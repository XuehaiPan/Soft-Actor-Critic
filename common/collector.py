import itertools
import os
import time
from functools import lru_cache, partialmethod

import numpy as np
import torch.multiprocessing as mp
import tqdm
from PIL import Image, ImageDraw
from setproctitle import setproctitle
from torch.utils.tensorboard import SummaryWriter

from .buffer import ReplayBuffer, EpisodeReplayBuffer
from .utils import clone_network, sync_params


__all__ = ['Collector', 'EpisodeCollector']


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, lock,
                 running_event, event, next_sampler_event,
                 env_func, env_kwargs, state_encoder, actor,
                 eval_only, replay_buffer,
                 n_total_steps, episode_steps, episode_rewards,
                 n_episodes, max_episode_steps,
                 deterministic, random_sample, render, log_episode_video,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}', daemon=True)

        self.rank = rank
        self.lock = lock
        self.running_event = running_event
        self.event = event
        self.next_sampler_event = next_sampler_event
        self.timeout = 60.0 * n_samplers

        self.env = None
        self.env_func = env_func
        self.env_kwargs = env_kwargs
        self.random_seed = random_seed

        self.shared_state_encoder = state_encoder
        self.shared_actor = actor
        self.state_encoder = None
        self.actor = None
        self.device = device
        self.eval_only = eval_only

        self.replay_buffer = replay_buffer
        self.n_total_steps = n_total_steps
        self.episode_steps = episode_steps
        self.episode_rewards = episode_rewards

        if np.isinf(n_episodes):
            self.n_episodes = np.inf
        else:
            self.n_episodes = n_episodes // n_samplers
            if rank < n_episodes % n_samplers:
                self.n_episodes += 1
        self.max_episode_steps = max_episode_steps
        self.deterministic = deterministic
        self.random_sample = random_sample
        self.render_env = (render and rank == 0)
        self.log_episode_video = (log_episode_video and rank == 0)

        self.log_dir = log_dir

        self.episode = 0
        self.trajectory = []
        self.frames = []
        self.render()

    def run(self):
        setproctitle(title=self.name)

        self.env = self.env_func(**self.env_kwargs)
        self.env.seed(self.random_seed)

        if not self.random_sample:
            self.state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
            self.actor = clone_network(src_net=self.shared_actor, device=self.device)
            self.state_encoder.eval()
            self.actor.eval()

        self.episode = 0
        while self.episode < self.n_episodes:
            self.episode += 1

            if not (self.eval_only or self.random_sample):
                sync_params(src_net=self.shared_state_encoder, dst_net=self.state_encoder)
                sync_params(src_net=self.shared_actor, dst_net=self.actor)

            episode_reward = 0
            episode_steps = 0
            self.trajectory.clear()
            if self.state_encoder is not None:
                self.state_encoder.reset()
            observation = self.env.reset()
            self.render()
            self.frames.clear()
            self.save_frame(step=0, reward=np.nan, episode_reward=0.0)
            for step in range(self.max_episode_steps):
                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state = self.state_encoder.encode(observation)
                    action = self.actor.get_action(state, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)

                episode_reward += reward
                episode_steps += 1
                self.render()
                self.save_frame(step=episode_steps, reward=reward, episode_reward=episode_reward)
                self.add_transaction(observation, action, reward, next_observation, done)

                observation = next_observation

                if done:
                    break

            self.running_event.wait()
            self.event.wait(timeout=self.timeout)
            with self.lock:
                self.save_trajectory()
                self.n_total_steps.value += episode_steps
                self.episode_steps.append(episode_steps)
                self.episode_rewards.append(episode_reward)
            self.event.clear()
            self.next_sampler_event.set()
            if self.writer is not None:
                average_reward = episode_reward / episode_steps
                self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=self.episode)
                self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=self.episode)
                self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=self.episode)
                self.log_video()
                self.writer.flush()

        self.env.close()

        if self.writer is not None:
            self.writer.close()

    def add_transaction(self, observation, action, reward, next_observation, done):
        self.trajectory.append((observation, action, [reward], next_observation, [done]))

    def save_trajectory(self):
        self.replay_buffer.extend(self.trajectory)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass
        super().close()

    def render(self, mode='human', **kwargs):
        if self.render_env:
            try:
                return self.env.render(mode=mode, **kwargs)
            except Exception:
                pass

    def save_frame(self, step, reward, episode_reward):
        if not self.random_sample and self.log_episode_video and self.episode % 100 == 0:
            try:
                img = self.env.render(mode='rgb_array')
            except Exception:
                pass
            else:
                text = (f'step           = {step}\n'
                        f'reward         = {reward:+.3f}\n'
                        f'episode reward = {episode_reward:+.3f}')
                img = Image.fromarray(img, mode='RGB')
                draw = ImageDraw.Draw(img)
                draw.multiline_text(xy=(10, 10), text=text, fill=(255, 0, 0))
                img = np.asanyarray(img, dtype=np.uint8)
                self.frames.append(img)

    def log_video(self):
        if self.writer is not None and self.log_episode_video and self.episode % 100 == 0:
            try:
                video = np.stack(self.frames).transpose((0, 3, 1, 2))
                video = np.expand_dims(video, axis=0)
                self.writer.add_video(tag='sample/episode', vid_tensor=video, global_step=self.episode, fps=120)
            except ValueError:
                pass

    @property
    @lru_cache(maxsize=None)
    def writer(self):
        if not self.random_sample and self.log_dir is not None:
            return SummaryWriter(log_dir=os.path.join(self.log_dir, f'sampler_{self.rank}'), comment=f'sampler_{self.rank}')
        else:
            return None


class EpisodeSampler(Sampler):
    def add_transaction(self, observation, action, reward, next_observation, done):
        self.trajectory.append((observation, action, [reward], [done]))

    def save_trajectory(self):
        self.replay_buffer.push(*tuple(map(np.stack, zip(*self.trajectory))))


class CollectorBase(object):
    def __init__(self, env_func, env_kwargs, state_encoder, actor,
                 n_samplers, sampler, replay_buffer, buffer_capacity,
                 devices, random_seed):
        self.manager = mp.Manager()
        self.running_event = self.manager.Event()
        self.running_event.set()
        self.total_steps = self.manager.Value('L', 0)
        self.episode_steps = self.manager.list()
        self.episode_rewards = self.manager.list()
        self.lock = self.manager.Lock()

        self.state_encoder = state_encoder
        self.actor = actor
        self.eval_only = False

        self.n_samplers = n_samplers
        self.sampler = sampler
        self.replay_buffer = replay_buffer(capacity=buffer_capacity, initializer=self.manager.list,
                                           Value=self.manager.Value, Lock=self.manager.Lock)

        self.env_func = env_func
        self.env_kwargs = env_kwargs
        self.devices = [device for _, device in zip(range(n_samplers), itertools.cycle(devices))]
        self.random_seed = random_seed

        self.samplers = []

    @property
    def n_episodes(self):
        return len(self.episode_steps)

    @property
    def n_total_steps(self):
        return self.total_steps.value

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_episode_video=False, log_dir=None):
        self.resume()

        events = [self.manager.Event() for i in range(self.n_samplers)]
        for event in events:
            event.clear()
        events[0].set()

        for rank in range(self.n_samplers):
            sampler = self.sampler(rank, self.n_samplers, self.lock,
                                   self.running_event, events[rank], events[(rank + 1) % self.n_samplers],
                                   self.env_func, self.env_kwargs, self.state_encoder, self.actor,
                                   self.eval_only, self.replay_buffer,
                                   self.total_steps, self.episode_steps, self.episode_rewards,
                                   n_episodes, max_episode_steps,
                                   deterministic, random_sample, render, log_episode_video,
                                   self.devices[rank], self.random_seed + rank, log_dir)
            sampler.start()
            self.samplers.append(sampler)

        return self.samplers

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_episode_video=False, log_dir=None):
        n_initial_episodes = self.n_episodes

        self.async_sample(n_episodes, max_episode_steps, deterministic, random_sample,
                          render, log_episode_video, log_dir)

        pbar = tqdm.tqdm(total=n_episodes, desc='Sampling')
        while True:
            n_new_episodes = self.n_episodes - n_initial_episodes
            if n_new_episodes > pbar.n:
                pbar.n = n_new_episodes
                pbar.set_postfix({'buffer_size': self.replay_buffer.size})
                if pbar.n >= n_episodes:
                    break
            else:
                time.sleep(0.1)

        self.join()

    def join(self):
        for sampler in self.samplers:
            sampler.join()
            sampler.close()
        self.samplers.clear()

    def terminate(self):
        self.pause()
        for sampler in self.samplers:
            if sampler.is_alive():
                try:
                    sampler.terminate()
                except Exception:
                    pass
        self.join()

    def pause(self):
        self.running_event.clear()

    def resume(self):
        self.running_event.set()

    def train(self, mode=True):
        self.eval_only = (not mode)
        return self

    def eval(self):
        return self.train(mode=False)


class Collector(CollectorBase):
    __init__ = partialmethod(CollectorBase.__init__,
                             sampler=Sampler,
                             replay_buffer=ReplayBuffer)


class EpisodeCollector(CollectorBase):
    __init__ = partialmethod(CollectorBase.__init__,
                             sampler=EpisodeSampler,
                             replay_buffer=EpisodeReplayBuffer)
