import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.nn.utils.rnn import pad_sequence

from common.buffer import TrajectoryReplayBuffer
from sac.rnn.network import SoftQNetwork, PolicyNetwork
from sac.trainer import Trainer as OriginTrainer


class Trainer(OriginTrainer):
    def __init__(self, env, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                 soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 buffer_capacity, writer, device):
        self.env = env
        self.device = device
        self.replay_buffer = TrajectoryReplayBuffer(capacity=buffer_capacity)
        self.n_episodes = 0
        self.episode_steps = []
        self.total_steps = 0
        self.writer = writer

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                         activation=F.relu, device=device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                         activation=F.relu, device=device)
        self.target_soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                                hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                                activation=F.relu, device=device)
        self.target_soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                                hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                                activation=F.relu, device=device)
        self.target_soft_q_net_1.load_state_dict(self.soft_q_net_1.state_dict())
        self.target_soft_q_net_2.load_state_dict(self.soft_q_net_2.state_dict())

        self.policy_net = PolicyNetwork(state_dim, action_dim,
                                        hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                        activation=F.relu, device=device)

        self.log_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device))

        self.modules = nn.ModuleDict({
            'soft_q_net_1': self.soft_q_net_1,
            'soft_q_net_2': self.soft_q_net_2,
            'target_soft_q_net_1': self.target_soft_q_net_1,
            'target_soft_q_net_2': self.target_soft_q_net_2,
            'policy_net': self.policy_net,
            'params': nn.ParameterDict({'log_alpha': self.log_alpha})
        })

        self.soft_q_criterion_1 = nn.MSELoss()
        self.soft_q_criterion_2 = nn.MSELoss()

        self.soft_q_optimizer = optim.Adam(itertools.chain(self.soft_q_net_1.parameters(), self.soft_q_net_2.parameters()),
                                           lr=soft_q_lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=weight_decay)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def env_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False):
        with tqdm.trange(n_episodes * max_episode_steps, desc='Sampling') as pbar:
            for episode in range(n_episodes):
                episode_reward = 0
                episode_steps = 0
                trajectory = []
                hidden = None
                state = self.env.reset()
                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                for step in range(max_episode_steps):
                    if random_sample:
                        action = self.policy_net.sample_action()
                    else:
                        action, hidden = self.policy_net.get_action(state, hidden, deterministic=deterministic)
                    next_state, reward, done, _ = self.env.step(action)
                    if render:
                        try:
                            self.env.render()
                        except Exception:
                            pass

                    episode_reward += reward
                    episode_steps += 1
                    trajectory.append((state, action, [reward], next_state, [done]))
                    state = next_state
                    if done:
                        pbar.update(max_episode_steps - step)
                        break
                    else:
                        pbar.update()
                self.replay_buffer.push(*tuple(map(np.stack, zip(*trajectory))))
                self.n_episodes += 1
                self.episode_steps.append(episode_steps)
                self.total_steps += episode_steps
                average_reward = episode_reward / episode_steps
                self.writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=self.n_episodes)
                self.writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=self.n_episodes)
                self.writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=self.n_episodes)

                pbar.set_postfix({'buffer_size': len(self.replay_buffer)})

    def update(self, batch_size, normalize_reward=True, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        # size: (batch, seq_len, item_size)
        batch_trajectory_state, batch_trajectory_action, batch_trajectory_reward, \
        batch_trajectory_next_state, batch_trajectory_done = self.replay_buffer.sample(batch_size)

        # size: (batch, 1, item_size)
        first_state = torch.stack(list(next(zip(*batch_trajectory_state)))).unsqueeze(dim=0).to(self.device)
        first_action = torch.stack(list(next(zip(*batch_trajectory_action)))).unsqueeze(dim=0).to(self.device)

        # size: (seq_len, batch, item_size)
        state = pad_sequence(batch_trajectory_state).to(self.device)
        next_state = pad_sequence(batch_trajectory_next_state).to(self.device)
        action = pad_sequence(batch_trajectory_action).to(self.device)
        reward = pad_sequence(batch_trajectory_reward).to(self.device)
        done = pad_sequence(batch_trajectory_done).to(self.device)

        # Normalize rewards
        if normalize_reward:
            reward = (reward - reward.mean()) / (reward.std() + epsilon)

        # Update temperature parameter
        new_action, log_prob, _ = self.policy_net.evaluate(state, None)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1.0

        # Training Q function
        predicted_q_value_1, _ = self.soft_q_net_1(state, action, None)
        predicted_q_value_2, _ = self.soft_q_net_2(state, action, None)
        with torch.no_grad():
            _, _, policy_net_second_state_hidden = self.policy_net.evaluate(first_state, None)
            new_next_action, next_log_prob, _ = self.policy_net.evaluate(next_state, policy_net_second_state_hidden)

            _, target_soft_q_net_1_second_state_hidden = self.target_soft_q_net_1(first_state, first_action, None)
            _, target_soft_q_net_2_second_state_hidden = self.target_soft_q_net_2(first_state, first_action, None)
            target_q_value_1, _ = self.target_soft_q_net_1(next_state, new_next_action, target_soft_q_net_1_second_state_hidden)
            target_q_value_2, _ = self.target_soft_q_net_1(next_state, new_next_action, target_soft_q_net_2_second_state_hidden)
            target_q_min = torch.min(target_q_value_1, target_q_value_2)
            target_q_min -= alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min
        q_value_loss_1 = self.soft_q_criterion_1(predicted_q_value_1, target_q_value)
        q_value_loss_2 = self.soft_q_criterion_2(predicted_q_value_2, target_q_value)

        self.soft_q_optimizer.zero_grad()
        q_value_loss_1.backward()
        q_value_loss_2.backward()
        self.soft_q_optimizer.step()

        # Training policy function
        predicted_new_q_value = torch.min(self.soft_q_net_1(state, new_action, None)[0],
                                          self.soft_q_net_2(state, new_action, None)[0])
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net_1.parameters(), self.soft_q_net_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net_2.parameters(), self.soft_q_net_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        return q_value_loss_1.item(), q_value_loss_2.item(), policy_loss.item()
