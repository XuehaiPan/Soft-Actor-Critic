import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.collector import TrajectoryCollector
from common.utils import sync_params
from sac.model import ModelBase as OriginalModelBase
from sac.rnn.network import SoftQNetwork, PolicyNetwork, StateEncoderWrapper


__all__ = ['ModelBase', 'Trainer', 'Tester']


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

        self.state_encoder = StateEncoderWrapper(state_encoder, device=self.model_device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                         skip_connection, activation=activation, device=self.model_device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                         activation=activation, device=self.model_device)
        self.policy_net = PolicyNetwork(state_dim, action_dim,
                                        hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                        activation=activation, device=self.model_device)

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

        self.collector = TrajectoryCollector(state_encoder=self.state_encoder,
                                             policy_net=self.policy_net,
                                             env=env,
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

    def update(self, batch_size, step_size=16,
               normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
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
                reward = reward_scale * (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob, _ = self.policy_net.evaluate(state, first_hidden)
        if adaptive_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            alpha = self.log_alpha.exp()

        # Train Q function
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

        # Train policy function
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
