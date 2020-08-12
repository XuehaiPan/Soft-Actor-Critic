import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.collector import Collector
from common.network import Container
from common.utils import clone_network, sync_params, init_optimizer, clip_grad_norm
from .network import StateEncoderWrapper, Actor, Critic


__all__ = ['build_model', 'Trainer', 'Tester']


def build_model(config):
    model_kwargs = config.build_from_keys(['env_func',
                                           'env_kwargs',
                                           'state_encoder',
                                           'state_dim',
                                           'action_dim',
                                           'hidden_dims',
                                           'activation',
                                           'initial_alpha',
                                           'n_samplers',
                                           'buffer_capacity',
                                           'devices',
                                           'random_seed'])
    if config.mode == 'train':
        model_kwargs.update(config.build_from_keys(['critic_lr',
                                                    'actor_lr',
                                                    'alpha_lr',
                                                    'weight_decay']))

        if not config.RNN_encoder:
            Model = Trainer
        else:
            from .rnn.model import Trainer as Model
    else:
        if not config.RNN_encoder:
            Model = Tester
        else:
            from .rnn.model import Tester as Model

    model = Model(**model_kwargs)
    model.print_info()
    for directory in (config.log_dir, config.checkpoint_dir):
        with open(file=os.path.join(directory, 'info.txt'), mode='w') as file:
            model.print_info(file=file)

    if config.initial_checkpoint is not None:
        model.load_model(path=config.initial_checkpoint)

    return model


class ModelBase(object):
    STATE_ENCODER_WRAPPER = StateEncoderWrapper
    COLLECTOR = Collector

    def __init__(self, env_func, env_kwargs, state_encoder,
                 state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, n_samplers, buffer_capacity,
                 devices, random_seed=0):
        self.devices = itertools.cycle(devices)
        self.model_device = next(self.devices)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = self.STATE_ENCODER_WRAPPER(state_encoder)

        self.critic = Critic(state_dim, action_dim, hidden_dims, activation=activation)
        self.actor = Actor(state_dim, action_dim, hidden_dims, activation=activation)

        self.log_alpha = nn.Parameter(torch.tensor(np.log(initial_alpha), dtype=torch.float32),
                                      requires_grad=True)

        self.modules = Container()
        self.modules.state_encoder = self.state_encoder
        self.modules.critic = self.critic
        self.modules.actor = self.actor
        self.modules.params = nn.ParameterDict({'log_alpha': self.log_alpha})
        self.modules.to(self.model_device)

        self.state_encoder.share_memory()
        self.actor.share_memory()
        self.collector = self.COLLECTOR(env_func=env_func,
                                        env_kwargs=env_kwargs,
                                        state_encoder=self.state_encoder,
                                        actor=self.actor,
                                        n_samplers=n_samplers,
                                        buffer_capacity=buffer_capacity,
                                        devices=self.devices,
                                        random_seed=random_seed)

    def print_info(self, file=None):
        print(f'state_dim = {self.state_dim}', file=file)
        print(f'action_dim = {self.action_dim}', file=file)
        print(f'device = {self.model_device}', file=file)
        print(f'buffer_capacity = {self.replay_buffer.capacity}', file=file)
        print(f'n_samplers = {self.collector.n_samplers}', file=file)
        print(f'sampler_devices = {list(map(str, self.collector.devices))}', file=file)
        print('Modules:', self.modules, file=file)

    @property
    def replay_buffer(self):
        return self.collector.replay_buffer

    def sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
               render=False, log_episode_video=False, log_dir=None):
        self.collector.sample(n_episodes, max_episode_steps, deterministic, random_sample,
                              render, log_episode_video, log_dir)

    def async_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False,
                     render=False, log_episode_video=False, log_dir=None):
        samplers = self.collector.async_sample(n_episodes, max_episode_steps,
                                               deterministic, random_sample,
                                               render, log_episode_video, log_dir)
        return samplers

    def train(self, mode=True):
        if self.training != mode:
            self.training = mode
            self.modules.train(mode=mode)
        self.collector.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def save_model(self, path):
        self.modules.save_model(path)

    def load_model(self, path, strict=True):
        self.modules.load_model(path, strict=strict)


class Trainer(ModelBase):
    def __init__(self, env_func, env_kwargs, state_encoder,
                 state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, critic_lr, actor_lr, alpha_lr, weight_decay,
                 n_samplers, buffer_capacity, devices, random_seed=0):
        super().__init__(env_func, env_kwargs, state_encoder,
                         state_dim, action_dim, hidden_dims, activation,
                         initial_alpha, n_samplers, buffer_capacity,
                         devices, random_seed)

        self.target_critic = clone_network(src_net=self.critic, device=self.model_device)
        self.target_critic.eval().requires_grad_(False)

        self.critic_criterion = nn.MSELoss()

        self.global_step = 0

        self.optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                    self.critic.parameters(),
                                                    self.actor.parameters()),
                                    lr=critic_lr, weight_decay=weight_decay)
        init_optimizer(self.optimizer)
        self.actor_loss_weight = actor_lr / critic_lr
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.train(mode=True)

    def update_sac(self, state, action, reward, next_state, done,
                   normalize_rewards=True, reward_scale=1.0,
                   adaptive_entropy=True, target_entropy=-2.0,
                   clip_gradient=False, gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        # Normalize rewards
        if normalize_rewards:
            with torch.no_grad():
                reward = reward_scale * (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob, _ = self.actor.evaluate(state)
        if adaptive_entropy:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Train Q function
        predicted_q_value_1, predicted_q_value_2 = self.critic(state, action)
        with torch.no_grad():
            new_next_action, next_log_prob, _ = self.actor.evaluate(next_state)

            target_q_min = torch.min(*self.target_critic(next_state, new_next_action))
            target_q_min -= alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min
        critic_loss_1 = self.critic_criterion(predicted_q_value_1, target_q_value)
        critic_loss_2 = self.critic_criterion(predicted_q_value_2, target_q_value)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2.0

        # Train policy function
        predicted_new_q_value = torch.min(*self.critic(state, new_action))
        predicted_new_q_value_critic_grad_only = torch.min(*self.critic(state, new_action.detach()))
        actor_loss = (alpha * log_prob - predicted_new_q_value).mean()
        actor_loss_unbiased = actor_loss + predicted_new_q_value_critic_grad_only.mean()

        loss = critic_loss + self.actor_loss_weight * actor_loss_unbiased
        self.optimizer.zero_grad()
        loss.backward()
        if clip_gradient:
            clip_grad_norm(self.optimizer)
        self.optimizer.step()

        # Soft update the target value net
        sync_params(src_net=self.critic, dst_net=self.target_critic, soft_tau=soft_tau)

        self.global_step += 1

        info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'temperature_parameter': alpha.item()
        }
        return info

    def update(self, batch_size, normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
               clip_gradient=False, gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        self.train()

        # size: (batch_size, item_size)
        state, action, reward, next_state, done = self.prepare_batch(batch_size)

        return self.update_sac(state, action, reward, next_state, done,
                               normalize_rewards, reward_scale,
                               adaptive_entropy, target_entropy,
                               clip_gradient, gamma, soft_tau, epsilon)

    def prepare_batch(self, batch_size):
        # size: (batch_size, item_size)
        observation, action, reward, next_observation, done \
            = tuple(map(lambda tensor: torch.FloatTensor(tensor).to(self.model_device),
                        self.replay_buffer.sample(batch_size)))

        state = self.state_encoder(observation)
        with torch.no_grad():
            next_state = self.state_encoder(next_observation)

        # size: (batch_size, item_size)
        return state, action, reward, next_state, done

    def load_model(self, path, strict=True):
        super().load_model(path=path, strict=strict)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval().requires_grad_(False)


class Tester(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval()
        self.modules.requires_grad_(False)
