import itertools
import os
import time
from functools import lru_cache

import numpy as np
import torch.multiprocessing as mp
import tqdm
from setproctitle import setproctitle
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.network import cat_hidden
from common.utils import clone_network, sync_params


__all__ = ['Collector', 'TrajectoryCollector']


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, lock,
                 running_event, event, next_sampler_event,
                 env_func, env_kwargs, state_encoder, actor,
                 eval_only, replay_buffer,
                 n_total_steps, episode_steps, episode_rewards,
                 n_episodes, max_episode_steps,
                 deterministic, random_sample, render, log_episode_video,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}')

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
        self.frames = []
        self.render()

    def run(self):
        setproctitle(title=self.name)

        self.env = self.env_func(**self.env_kwargs)
        self.env.seed(self.random_seed)

        if not self.random_sample:
            state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
            actor = clone_network(src_net=self.shared_actor, device=self.device)
            state_encoder.eval()
            actor.eval()
        else:
            state_encoder = actor = None

        self.episode = 0
        while self.episode < self.n_episodes:
            self.episode += 1

            if not (self.eval_only or self.random_sample):
                sync_params(src_net=self.shared_state_encoder, dst_net=state_encoder)
                sync_params(src_net=self.shared_actor, dst_net=actor)

            episode_reward = 0
            episode_steps = 0
            trajectory = []
            observation = self.env.reset()
            self.render()
            self.frames.clear()
            self.save_frame()
            for step in range(self.max_episode_steps):
                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state = state_encoder.encode(observation)
                    action = actor.get_action(state, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)
                self.render()
                self.save_frame()

                episode_reward += reward
                episode_steps += 1
                trajectory.append((observation, action, [reward], next_observation, [done]))
                observation = next_observation

                if done:
                    break

            self.running_event.wait()
            self.event.wait(timeout=self.timeout)
            with self.lock:
                self.replay_buffer.extend(trajectory)
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

    def save_frame(self):
        if not self.random_sample and self.log_episode_video and self.episode % 100 == 0:
            try:
                self.frames.append(self.env.render(mode='rgb_array'))
            except Exception:
                pass

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


class TrajectorySampler(Sampler):
    def run(self):
        setproctitle(title=self.name)

        self.env = self.env_func(**self.env_kwargs)
        self.env.seed(self.random_seed)

        state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
        actor = clone_network(src_net=self.shared_actor, device=self.device)
        state_encoder.eval()
        actor.eval()

        self.episode = 0
        while self.episode < self.n_episodes:
            self.episode += 1

            if not (self.eval_only or self.random_sample):
                sync_params(src_net=self.shared_state_encoder, dst_net=state_encoder)
                sync_params(src_net=self.shared_actor, dst_net=actor)

            episode_reward = 0
            episode_steps = 0
            trajectory = []
            hiddens = []
            hidden = state_encoder.initial_hiddens(batch_size=1)
            observation = self.env.reset()
            self.frames.clear()
            self.render()
            self.save_frame()
            for step in range(self.max_episode_steps):
                hiddens.append(hidden)

                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state, hidden = state_encoder.encode(observation, hidden=hidden)
                    action = actor.get_action(state, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)
                self.render()
                self.save_frame()

                episode_reward += reward
                episode_steps += 1
                trajectory.append((observation, action, [reward], next_observation, [done]))
                observation = next_observation

                if done:
                    break

            hiddens = cat_hidden(hiddens, dim=0).cpu().detach().numpy()

            self.running_event.wait()
            self.event.wait(timeout=self.timeout)
            with self.lock:
                self.replay_buffer.push(*tuple(map(np.stack, zip(*trajectory))), hiddens)
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

        samplers = []
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
            samplers.append(sampler)

        return samplers

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_episode_video=False, log_dir=None):
        n_initial_episodes = self.n_episodes

        samplers = self.async_sample(n_episodes, max_episode_steps, deterministic, random_sample,
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

        for sampler in samplers:
            sampler.join()
            sampler.close()

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
    def __init__(self, env_func, env_kwargs, state_encoder, actor,
                 n_samplers, buffer_capacity, devices, random_seed):
        super().__init__(env_func, env_kwargs,
                         state_encoder, actor,
                         n_samplers, Sampler,
                         ReplayBuffer, buffer_capacity,
                         devices, random_seed)


class TrajectoryCollector(CollectorBase):
    def __init__(self, env_func, env_kwargs, state_encoder,
                 actor, n_samplers, buffer_capacity,
                 devices, random_seed):
        super().__init__(env_func, env_kwargs,
                         state_encoder, actor,
                         n_samplers, TrajectorySampler,
                         TrajectoryReplayBuffer, buffer_capacity,
                         devices, random_seed)
