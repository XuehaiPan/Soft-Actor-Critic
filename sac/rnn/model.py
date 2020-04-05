import copy
import itertools
import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.utils.tensorboard import SummaryWriter

from common.buffer import TrajectoryReplayBuffer
from common.utils import sync_params
from sac.model import Collector as OriginalCollector, ModelBase as OriginalModelBase
from sac.network import SoftQNetwork, PolicyNetwork, EncoderWrapper
from sac.rnn.network import SoftQNetwork, PolicyNetwork, cat_hidden, EncoderWrapper


__all__ = ['Collector', 'ModelBase', 'Trainer', 'Tester']


class Collector(OriginalCollector):
    def __init__(self, model, env, buffer_capacity, n_samplers, devices, random_seed=None):
        super().__init__(model, env, buffer_capacity, n_samplers, devices, random_seed)

        self.replay_buffer = TrajectoryReplayBuffer(capacity=buffer_capacity, initializer=self.manager.list, lock=self.manager.Lock())

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
                hiddens = []
                hidden = policy_net_local.initial_hiddens(batch_size=1)
                observation = env_local.reset()
                if render:
                    try:
                        env_local.render()
                    except Exception:
                        pass
                for step in range(max_episode_steps):
                    hiddens.append(hidden)

                    if random_sample:
                        action = env_local.action_space.sample()
                    else:
                        state = state_encoder_local.encode(observation)
                        action, hidden = policy_net_local.get_action(state, hidden, deterministic=deterministic)
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
                hiddens = cat_hidden(hiddens, dim=0).detach().cpu()
                cache.put((episode_steps, episode_reward, trajectory, hiddens))
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

        cache = self.manager.Queue(maxsize=2 * self.n_samplers)

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
            episode_steps, episode_reward, trajectory, hiddens = items
            self.replay_buffer.push(*tuple(map(np.stack, zip(*trajectory))), hiddens)
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


class ModelBase(OriginalModelBase):
    def __init__(self, env, state_encoder, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                 skip_connection, activation,
                 initial_alpha, n_samplers, buffer_capacity, devices, random_seed=None):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = EncoderWrapper(state_encoder, device=self.model_device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                         skip_connection, activation=activation, device=self.model_device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                         activation=activation, device=self.model_device)
        self.policy_net = PolicyNetwork(state_dim, action_dim,
                                        hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                        activation=F.relu, device=self.model_device)

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


class Trainer(ModelBase):
    def __init__(self, env, state_encoder, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                 skip_connection, activation,
                 initial_alpha, soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 n_samplers, buffer_capacity, devices, random_seed=None):
        super().__init__(env, state_encoder, state_dim, action_dim,
                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                         skip_connection, activation,
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

    def update(self, batch_size, step_size=16, normalize_rewards=True, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        self.train()

        # size: (seq_len, batch_size, item_size)
        observation, action, reward, next_observation, done, hidden = tuple(map(lambda tensor: tensor.to(self.model_device),
                                                                                self.replay_buffer.sample(batch_size, step_size=step_size)))

        state = self.state_encoder(observation)
        with torch.no_grad():
            next_state = self.state_encoder(next_observation)

        # size: (1, batch_size, item_size)
        first_state = state[0].unsqueeze(dim=0)
        first_action = action[0].unsqueeze(dim=0)
        first_hidden = hidden[0].unsqueeze(dim=0)

        # Normalize rewards
        if normalize_rewards:
            with torch.no_grad():
                reward = (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob, _ = self.policy_net.evaluate(state, first_hidden)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Training Q function
        predicted_q_value_1, _ = self.soft_q_net_1(state, action, hidden)
        predicted_q_value_2, _ = self.soft_q_net_2(state, action, hidden)
        with torch.no_grad():
            _, _, policy_net_second_hidden = self.policy_net.evaluate(first_state, first_hidden)
            new_next_action, next_log_prob, _ = self.policy_net.evaluate(next_state, policy_net_second_hidden)

            _, target_soft_q_net_1_second_hidden = self.target_soft_q_net_1(first_state, first_action, first_hidden)
            _, target_soft_q_net_2_second_hidden = self.target_soft_q_net_2(first_state, first_action, first_hidden)
            target_q_value_1, _ = self.target_soft_q_net_1(next_state, new_next_action, target_soft_q_net_1_second_hidden)
            target_q_value_2, _ = self.target_soft_q_net_1(next_state, new_next_action, target_soft_q_net_2_second_hidden)
            target_q_min = torch.min(target_q_value_1, target_q_value_2)
            target_q_min -= alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min
        soft_q_loss_1 = self.soft_q_criterion_1(predicted_q_value_1, target_q_value)
        soft_q_loss_2 = self.soft_q_criterion_2(predicted_q_value_2, target_q_value)
        soft_q_loss = (soft_q_loss_1 + soft_q_loss_2) / 2.0

        # Training policy function
        predicted_new_q_value = torch.min(self.soft_q_net_1(state, new_action, hidden)[0],
                                          self.soft_q_net_2(state, new_action, hidden)[0])
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
