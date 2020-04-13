import copy
import itertools
import os
from functools import lru_cache

import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer, TrajectoryReplayBuffer
from common.utils import clone_network, sync_params
from sac.rnn.network import cat_hidden


__all__ = ['Collector', 'TrajectoryCollector']


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, lock,
                 running_event, event, next_sampler_event,
                 env, state_encoder, policy_net,
                 replay_buffer, episode_steps, episode_rewards,
                 n_episodes, max_episode_steps,
                 deterministic, random_sample, render,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}')

        self.rank = rank
        self.lock = lock
        self.running_event = running_event
        self.event = event
        self.next_sampler_event = next_sampler_event

        self.env = copy.deepcopy(env)
        self.env.seed(random_seed)

        self.shared_state_encoder = state_encoder
        self.shared_policy_net = policy_net
        self.device = device

        self.replay_buffer = replay_buffer
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

        self.log_dir = log_dir

    def run(self):
        state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
        policy_net = clone_network(src_net=self.shared_policy_net, device=self.device)
        state_encoder.eval()
        policy_net.eval()

        episode = 0
        while episode < self.n_episodes:
            sync_params(src_net=self.shared_state_encoder, dst_net=state_encoder)
            sync_params(src_net=self.shared_policy_net, dst_net=policy_net)

            episode_reward = 0
            episode_steps = 0
            trajectory = []
            observation = self.env.reset()
            self.render()
            for step in range(self.max_episode_steps):
                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state = state_encoder.encode(observation)
                    action = policy_net.get_action(state, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)
                self.render()

                episode_reward += reward
                episode_steps += 1
                trajectory.append((observation, action, [reward], next_observation, [done]))
                observation = next_observation

            self.running_event.wait()
            self.event.wait()
            with self.lock:
                self.replay_buffer.extend(trajectory)
                self.episode_steps.append(episode_steps)
                self.episode_rewards.append(episode_reward)
            self.event.clear()
            self.next_sampler_event.set()
            episode += 1
            if self.writer is not None:
                average_reward = episode_reward / episode_steps
                self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=episode)
                self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=episode)
                self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=episode)
                self.writer.flush()

        if self.writer is not None:
            self.writer.close()

    def render(self):
        if self.render_env:
            try:
                self.env.render()
            except Exception:
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
        state_encoder = clone_network(src_net=self.shared_state_encoder, device=self.device)
        policy_net = clone_network(src_net=self.shared_policy_net, device=self.device)
        state_encoder.eval()
        policy_net.eval()

        episode = 0
        while episode < self.n_episodes:
            sync_params(src_net=self.shared_state_encoder, dst_net=state_encoder)
            sync_params(src_net=self.shared_policy_net, dst_net=policy_net)

            episode_reward = 0
            episode_steps = 0
            trajectory = []
            hiddens = []
            hidden = policy_net.initial_hiddens(batch_size=1)
            observation = self.env.reset()
            self.render()
            for step in range(self.max_episode_steps):
                hiddens.append(hidden)

                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state = state_encoder.encode(observation)
                    action, hidden = policy_net.get_action(state, hidden, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)
                self.render()

                episode_reward += reward
                episode_steps += 1
                trajectory.append((observation, action, [reward], next_observation, [done]))
                observation = next_observation
            hiddens = cat_hidden(hiddens, dim=0).detach().cpu()

            self.running_event.wait()
            self.event.wait()
            with self.lock:
                self.replay_buffer.push(*tuple(map(np.stack, zip(*trajectory))), hiddens)
                self.episode_steps.append(episode_steps)
                self.episode_rewards.append(episode_reward)
            self.event.clear()
            self.next_sampler_event.set()
            episode += 1
            if self.writer is not None:
                average_reward = episode_reward / episode_steps
                self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=episode)
                self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=episode)
                self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=episode)
                self.writer.flush()

        if self.writer is not None:
            self.writer.close()


class CollectorBase(object):
    def __init__(self, state_encoder, policy_net, sampler, replay_buffer,
                 env, buffer_capacity, n_samplers, devices, random_seed):
        self.manager = mp.Manager()
        self.running_event = self.manager.Event()
        self.running_event.set()
        self.episode_steps = self.manager.list()
        self.episode_rewards = self.manager.list()
        self.lock = self.manager.Lock()

        self.state_encoder = state_encoder
        self.policy_net = policy_net
        self.replay_buffer = replay_buffer(capacity=buffer_capacity, initializer=self.manager.list, lock=self.manager.Lock())

        self.n_samplers = n_samplers
        self.sampler = sampler

        self.env = env
        self.devices = [device for _, device in zip(range(n_samplers), itertools.cycle(devices))]
        self.random_seed = random_seed

    @property
    def n_episodes(self):
        return len(self.episode_steps)

    @property
    def total_steps(self):
        return sum(self.episode_steps)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_dir=None):
        self.resume()

        events = [self.manager.Event() for i in range(self.n_samplers)]
        for event in events:
            event.clear()
        events[0].set()

        samplers = []
        for rank in range(self.n_samplers):
            sampler = self.sampler(rank, self.n_samplers, self.lock,
                                   self.running_event, events[rank], events[(rank + 1) % self.n_samplers],
                                   self.env, self.state_encoder, self.policy_net,
                                   self.replay_buffer, self.episode_steps, self.episode_rewards,
                                   n_episodes, max_episode_steps,
                                   deterministic, random_sample, render,
                                   self.devices[rank], self.random_seed + rank, log_dir)
            sampler.start()
            samplers.append(sampler)

        return samplers

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_dir=None):
        samplers = self.async_sample(n_episodes, max_episode_steps, deterministic, random_sample,
                                     render, log_dir)

        for sampler in samplers:
            sampler.join()

    def pause(self):
        self.running_event.clear()

    def resume(self):
        self.running_event.set()


class Collector(CollectorBase):
    def __init__(self, state_encoder, policy_net,
                 env, buffer_capacity, n_samplers, devices, random_seed):
        super().__init__(state_encoder, policy_net, Sampler, ReplayBuffer,
                         env, buffer_capacity, n_samplers, devices, random_seed)


class TrajectoryCollector(CollectorBase):
    def __init__(self, state_encoder, policy_net,
                 env, buffer_capacity, n_samplers, devices, random_seed):
        super().__init__(state_encoder, policy_net, TrajectorySampler, TrajectoryReplayBuffer,
                         env, buffer_capacity, n_samplers, devices, random_seed)
