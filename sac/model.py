import copy
import itertools
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer
from common.utils import sync_params
from sac.network import SoftQNetwork, PolicyNetwork, StateEncoderWrapper


__all__ = ['Collector', 'ModelBase', 'Trainer', 'Tester']


class Sampler(mp.Process):
    def __init__(self, rank, n_samplers, lock, event, env, state_encoder, policy_net,
                 replay_buffer, episode_steps, episode_rewards,
                 n_episodes, max_episode_steps,
                 deterministic, random_sample, render,
                 device, random_seed, log_dir):
        super().__init__(name=f'sampler_{rank}')

        self.rank = rank
        self.n_samplers = n_samplers
        self.lock = lock
        self.event = event

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
        self.render = render

        self.log_dir = log_dir

    def run(self):
        if not self.random_sample and self.log_dir is not None:
            writer = SummaryWriter(log_dir=os.path.join(self.log_dir, f'sampler_{self.rank}'), comment=f'sampler_{self.rank}')
        else:
            writer = None

        state_encoder = copy.deepcopy(self.shared_state_encoder)
        policy_net = copy.deepcopy(self.shared_policy_net)
        state_encoder.device = self.device
        policy_net.device = self.device
        state_encoder.to(self.device)
        policy_net.to(self.device)

        episode = 0
        while episode < self.n_episodes:
            sync_params(src_net=self.shared_state_encoder, dst_net=state_encoder)
            sync_params(src_net=self.shared_policy_net, dst_net=policy_net)
            state_encoder.eval()
            policy_net.eval()

            episode_reward = 0
            episode_steps = 0
            trajectory = []
            observation = self.env.reset()
            if self.render:
                try:
                    self.env.render()
                except Exception:
                    pass
            for step in range(self.max_episode_steps):
                if self.random_sample:
                    action = self.env.action_space.sample()
                else:
                    state = state_encoder.encode(observation)
                    action = policy_net.get_action(state, deterministic=self.deterministic)
                next_observation, reward, done, _ = self.env.step(action)
                if self.render:
                    try:
                        self.env.render()
                    except Exception:
                        pass

                episode_reward += reward
                episode_steps += 1
                trajectory.append((observation, action, [reward], next_observation, [done]))
                observation = next_observation

            self.event.wait()
            with self.lock:
                self.replay_buffer.extend(trajectory)
                self.episode_steps.append(episode_steps)
                self.episode_rewards.append(episode_reward)
            episode += 1
            if writer is not None:
                average_reward = episode_reward / episode_steps
                writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=episode)
                writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=episode)
                writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=episode)
                writer.flush()

        if writer is not None:
            writer.close()


class Collector(object):
    def __init__(self, state_encoder, policy_net, sampler, replay_buffer, env, buffer_capacity, n_samplers, devices, random_seed):
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

        samplers = []
        for rank in range(self.n_samplers):
            sampler = self.sampler(rank, self.n_samplers, self.lock, self.running_event,
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


class ModelBase(object):
    def __init__(self, env, state_encoder, state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, n_samplers, buffer_capacity, devices, random_seed=0):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = StateEncoderWrapper(state_encoder, device=self.model_device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims,
                                         activation=activation, device=self.model_device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims,
                                         activation=activation, device=self.model_device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=self.model_device)

        self.log_alpha = nn.Parameter(torch.tensor([[np.log(initial_alpha)]], dtype=torch.float32, device=self.model_device),
                                      requires_grad=True)

        self.modules = nn.ModuleDict([
            ('state_encoder', self.state_encoder),
            ('soft_q_net_1', self.soft_q_net_1),
            ('soft_q_net_2', self.soft_q_net_2),
            ('policy_net', self.policy_net),
            ('params', nn.ParameterDict({'log_alpha': self.log_alpha}))
        ])
        self.modules.share_memory()

        self.collector = Collector(state_encoder=self.state_encoder,
                                   policy_net=self.policy_net,
                                   sampler=Sampler,
                                   replay_buffer=ReplayBuffer,
                                   env=env,
                                   buffer_capacity=buffer_capacity,
                                   n_samplers=n_samplers,
                                   devices=self.devices,
                                   random_seed=random_seed)

    def print_info(self):
        print(f'env = {self.env}')
        print(f'state_dim = {self.state_dim}')
        print(f'action_dim = {self.action_dim}')
        print(f'device = {self.model_device}')
        print(f'buffer_capacity = {self.replay_buffer.capacity}')
        print(f'n_samplers = {self.collector.n_samplers}')
        print(f'sampler_devices = {list(map(str, self.collector.devices))}')
        print('Modules:', self.modules)

    @property
    def env(self):
        return self.collector.env

    @property
    def replay_buffer(self):
        return self.collector.replay_buffer

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_dir=None):
        self.collector.sample(n_episodes, max_episode_steps, deterministic, random_sample,
                              render, log_dir)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_dir=None):
        samplers = self.collector.async_sample(n_episodes, max_episode_steps,
                                               deterministic, random_sample,
                                               render, log_dir)
        return samplers

    def save_model(self, path):
        torch.save(self.modules.state_dict(), path)

    def load_model(self, path):
        self.modules.load_state_dict(torch.load(path, map_location=self.model_device))


class Trainer(ModelBase):
    def __init__(self, env, state_encoder, state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 n_samplers, buffer_capacity, devices, random_seed=None):
        super().__init__(env, state_encoder, state_dim, action_dim, hidden_dims, activation,
                         initial_alpha, n_samplers, buffer_capacity, devices, random_seed)

        self.training = True

        self.target_soft_q_net_1 = copy.deepcopy(self.soft_q_net_1)
        self.target_soft_q_net_2 = copy.deepcopy(self.soft_q_net_2)
        self.target_soft_q_net_1.eval()
        self.target_soft_q_net_2.eval()

        self.soft_q_criterion_1 = nn.MSELoss()
        self.soft_q_criterion_2 = nn.MSELoss()

        self.optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                    self.soft_q_net_1.parameters(),
                                                    self.soft_q_net_2.parameters(),
                                                    self.policy_net.parameters()),
                                    lr=soft_q_lr, weight_decay=weight_decay)
        self.policy_loss_weight = policy_lr / soft_q_lr
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def update(self, batch_size, normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        self.train()

        # size: (batch_size, item_size)
        observation, action, reward, next_observation, done = tuple(map(lambda tensor: tensor.to(self.model_device),
                                                                        self.replay_buffer.sample(batch_size)))

        state = self.state_encoder(observation)
        with torch.no_grad():
            next_state = self.state_encoder(next_observation)

        # Normalize rewards
        if normalize_rewards:
            with torch.no_grad():
                reward = reward_scale * (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob = self.policy_net.evaluate(state)
        if adaptive_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Train Q function
        predicted_q_value_1 = self.soft_q_net_1(state, action)
        predicted_q_value_2 = self.soft_q_net_2(state, action)
        with torch.no_grad():
            new_next_action, next_log_prob = self.policy_net.evaluate(next_state)

            target_q_min = torch.min(self.target_soft_q_net_1(next_state, new_next_action),
                                     self.target_soft_q_net_2(next_state, new_next_action))
            target_q_min -= alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min
        soft_q_loss_1 = self.soft_q_criterion_1(predicted_q_value_1, target_q_value)
        soft_q_loss_2 = self.soft_q_criterion_2(predicted_q_value_2, target_q_value)
        soft_q_loss = (soft_q_loss_1 + soft_q_loss_2) / 2.0

        # Train policy function
        predicted_new_q_value = torch.min(self.soft_q_net_1(state, new_action),
                                          self.soft_q_net_2(state, new_action))
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        loss = soft_q_loss + self.policy_loss_weight * policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target value net
        sync_params(src_net=self.soft_q_net_1, dst_net=self.target_soft_q_net_1, soft_tau=soft_tau)
        sync_params(src_net=self.soft_q_net_2, dst_net=self.target_soft_q_net_2, soft_tau=soft_tau)

        info = {
            'action_scale': (self.soft_q_net_1.action_scale + self.soft_q_net_2.action_scale) / 2.0
        }
        return soft_q_loss.item(), policy_loss.item(), alpha.item(), info

    def train(self, mode=True):
        self.training = mode
        self.modules.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def load_model(self, path):
        super().load_model(path=path)
        self.target_soft_q_net_1.load_state_dict(self.soft_q_net_1.state_dict())
        self.target_soft_q_net_2.load_state_dict(self.soft_q_net_2.state_dict())
        self.target_soft_q_net_1.eval()
        self.target_soft_q_net_2.eval()


class Tester(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modules.eval()
