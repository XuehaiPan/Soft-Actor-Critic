import copy
import itertools
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.buffer import ReplayBuffer
from common.utils import sync_params
from sac.network import SoftQNetwork, PolicyNetwork, EncoderWrapper


__all__ = ['Collector', 'ModelBase', 'Trainer', 'Tester']


class Collector(object):
    def __init__(self, model, env, buffer_capacity, n_samplers, devices, random_seed=None):
        self.manager = mp.Manager()
        self.running_event = self.manager.Event()
        self.running_event.set()
        self.episode_steps = self.manager.list()
        self.episode_rewards = self.manager.list()
        self.offset = 0

        self.model = model
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, initializer=self.manager.list, lock=self.manager.Lock())
        self.n_samplers = n_samplers

        self.env = env
        self.devices = devices
        self.random_seed = self.env.seed(random_seed)[0]

    @property
    def n_episodes(self):
        return len(self.episode_steps)

    @property
    def total_steps(self):
        return sum(self.episode_steps)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False, log_dir=None):
        collect_process = mp.Process(target=self.sample,
                                     args=(n_episodes, max_episode_steps, deterministic, random_sample, render, log_dir, False),
                                     name='collector')
        collect_process.start()

        return collect_process

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_dir=None, progress=False):
        def sampler_target(collector, cache, n_episodes, max_episode_steps,
                           deterministic, random_sample, render,
                           device, random_seed, log_dir):
            if not random_sample and log_dir is not None:
                sampler_writer = SummaryWriter(log_dir=log_dir, comment='sampler')
            else:
                sampler_writer = None

            env_local = copy.deepcopy(collector.env)
            env_local.seed(random_seed)
            state_encoder_local = copy.deepcopy(collector.model.state_encoder)
            policy_net_local = copy.deepcopy(collector.model.policy_net)
            if device is not None:
                state_encoder_local.to(device)
                policy_net_local.to(device)

            episode = 0
            while episode < n_episodes:
                sync_params(src_net=collector.model.state_encoder, dst_net=state_encoder_local, soft_tau=1.0)
                sync_params(src_net=collector.model.policy_net, dst_net=policy_net_local, soft_tau=1.0)
                state_encoder_local.eval()
                policy_net_local.eval()

                episode_reward = 0
                episode_steps = 0
                trajectory = []
                observation = env_local.reset()
                if render:
                    try:
                        env_local.render()
                    except Exception:
                        pass
                for step in range(max_episode_steps):
                    if random_sample:
                        action = env_local.action_space.sample()
                    else:
                        state = state_encoder_local.encode(observation)
                        action = policy_net_local.get_action(state, deterministic=deterministic)
                    next_observation, reward, done, _ = env_local.step(action)
                    if render:
                        try:
                            env_local.render()
                        except Exception:
                            pass

                    episode_reward += reward
                    episode_steps += 1
                    trajectory.append((observation, action, [reward], next_observation, [done]))
                    observation = next_observation
                cache.put((episode_steps, episode_reward, trajectory))
                episode += 1
                if sampler_writer is not None:
                    average_reward = episode_reward / episode_steps
                    sampler_writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=episode)
                    sampler_writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=episode)
                    sampler_writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=episode)
                    sampler_writer.flush()

            cache.put(None)
            if sampler_writer is not None:
                sampler_writer.close()

        cache = mp.Queue(maxsize=2 * self.n_samplers)

        sample_processes = []
        devices = self.devices
        if self.devices is None:
            devices = [None]
        for i, device in zip(range(self.n_samplers), itertools.cycle(devices)):
            if np.isinf(n_episodes):
                sampler_n_episodes = np.inf
            else:
                sampler_n_episodes = n_episodes // self.n_samplers
                if i < n_episodes % self.n_samplers:
                    sampler_n_episodes += 1
            if log_dir is not None:
                sampler_log_dir = os.path.join(log_dir, f'sampler_{i}')
            else:
                sampler_log_dir = None
            sample_process = mp.Process(target=sampler_target,
                                        args=(self, cache, sampler_n_episodes, max_episode_steps,
                                              deterministic, random_sample, render,
                                              device, self.random_seed + i, sampler_log_dir),
                                        name=f'sampler_{i}')
            sample_process.start()
            sample_processes.append(sample_process)

        if not random_sample and log_dir is not None:
            collector_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'collector'), comment='collector')
        else:
            collector_writer = None
        n_alive = self.n_samplers
        if progress and not np.isinf(n_episodes):
            pbar = tqdm.tqdm(total=n_episodes, desc='Sampling')
        else:
            pbar = None
        while n_alive > 0:
            self.running_event.wait()
            items = cache.get()
            if items is None:
                n_alive -= 1
                continue
            episode_steps, episode_reward, trajectory = items
            self.replay_buffer.extend(trajectory)
            self.episode_steps.append(episode_steps)
            self.episode_rewards.append(episode_reward)
            if pbar is not None:
                pbar.update()
            if collector_writer is not None:
                collector_writer.add_scalar(tag='sample/buffer_size', scalar_value=self.replay_buffer.size,
                                            global_step=self.n_episodes)

        if pbar is not None:
            pbar.close()

        for sample_process in sample_processes:
            sample_process.join()

    def pause(self):
        self.running_event.clear()

    def resume(self):
        self.running_event.set()


class ModelBase(object):
    def __init__(self, env, state_encoder, state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, n_samplers, buffer_capacity, devices, random_seed=None):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = EncoderWrapper(state_encoder, device=self.model_device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=self.model_device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=self.model_device)
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

        self.collector = Collector(model=self, env=env,
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
        print('Modules:', self.modules)

    @property
    def env(self):
        return self.collector.env

    @property
    def replay_buffer(self):
        return self.collector.replay_buffer

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False, log_dir=None, progress=False):
        self.collector.sample(n_episodes, max_episode_steps, deterministic, random_sample, render, log_dir, progress)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False, log_dir=None):
        collector_process = self.collector.async_sample(n_episodes, max_episode_steps,
                                                        deterministic, random_sample,
                                                        render, log_dir)
        return collector_process

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

    def update(self, batch_size, normalize_rewards=True, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        self.train()

        # size: (batch_size, item_size)
        observation, action, reward, next_observation, done = tuple(map(lambda tensor: tensor.to(self.model_device),
                                                                        self.replay_buffer.sample(batch_size)))

        state = self.state_encoder(observation)
        with torch.no_grad():
            next_state = self.state_encoder(next_observation)

        # Normalize rewards
        if normalize_rewards:
            reward = (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob = self.policy_net.evaluate(state)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Training Q function
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

        # Training policy function
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

        return soft_q_loss.item(), policy_loss.item(), alpha.item()

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
