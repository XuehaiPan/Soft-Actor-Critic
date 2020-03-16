import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from common.buffer import ReplayBuffer
from sac.network import SoftQNetwork, PolicyNetwork


class Trainer(object):
    def __init__(self, env, state_dim, action_dim, hidden_dims,
                 soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 buffer_capacity, writer, device):
        self.env = env
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.n_episodes = 0
        self.episode_steps = []
        self.total_steps = 0
        self.writer = writer

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)
        self.target_soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)
        self.target_soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims, activation=F.relu, device=device)

        self.target_soft_q_net_1.load_state_dict(self.soft_q_net_1.state_dict())
        self.target_soft_q_net_2.load_state_dict(self.soft_q_net_2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.log_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device))

        self.modules = nn.ModuleDict({
            'soft_q_net_1': self.soft_q_net_1,
            'soft_q_net_2': self.soft_q_net_2,
            'target_soft_q_net_1': self.target_soft_q_net_1,
            'target_soft_q_net_2': self.target_soft_q_net_2,
            'policy_net': self.policy_net,
            'target_policy_net': self.target_policy_net,
            'params': nn.ParameterDict({'log_alpha': self.log_alpha})
        })

        self.soft_q_criterion_1 = nn.MSELoss()
        self.soft_q_criterion_2 = nn.MSELoss()

        self.soft_q_optimizer = optim.Adam(itertools.chain(self.soft_q_net_1.parameters(), self.soft_q_net_2.parameters()),
                                           lr=soft_q_lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=weight_decay)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def print_info(self):
        print(f'env = {self.env}')
        print(f'state_dim = {self.state_dim}')
        print(f'action_dim = {self.action_dim}')
        print(f'device = {self.device}')
        print(f'buffer_capacity = {self.replay_buffer.capacity}')
        print('Modules:', self.modules)

    def env_sample(self, n_episodes, max_steps, deterministic, epsilon):
        with tqdm.trange(n_episodes * max_steps, desc='Sampling') as pbar:
            for episode in range(n_episodes):
                episode_reward = 0
                episode_step = 0
                state = self.env.reset()
                for step in range(max_steps):
                    if np.random.binomial(1, epsilon) != 0:
                        action = self.policy_net.sample_action()
                    else:
                        action = self.policy_net.get_action(state, deterministic=deterministic)
                    next_state, reward, done, _ = self.env.step(action)

                    episode_reward += reward
                    episode_step += 1
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        pbar.update(max_steps - step)
                        break
                    else:
                        pbar.update()
                self.n_episodes += 1
                self.episode_steps.append(episode_step)
                self.total_steps += episode_step
                self.writer.add_scalar(tag='sample/rewards', scalar_value=episode_reward, global_step=self.n_episodes)
                self.writer.add_scalar(tag='sample/steps', scalar_value=episode_step, global_step=self.n_episodes)

                pbar.set_postfix({'buffer_size': len(self.replay_buffer)})

    def update(self, batch_size, reward_scale=1.0, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(dim=1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(dim=1).to(self.device)

        predicted_q_value_1 = self.soft_q_net_1(state, action)
        predicted_q_value_2 = self.soft_q_net_2(state, action)
        new_action, log_prob = self.policy_net.evaluate(state)
        reward = reward_scale * (reward - reward.mean()) / (reward.std() + epsilon)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1.0

        # Training Q Function
        with torch.no_grad():
            new_next_action, next_log_prob = self.target_policy_net.evaluate(next_state)

            target_q_min = torch.min(self.target_soft_q_net_1(next_state, new_next_action),
                                     self.target_soft_q_net_2(next_state, new_next_action)) \
                           - alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * target_q_min
        q_value_loss_1 = self.soft_q_criterion_1(predicted_q_value_1, target_q_value)
        q_value_loss_2 = self.soft_q_criterion_2(predicted_q_value_2, target_q_value)

        self.soft_q_optimizer.zero_grad()
        q_value_loss_1.backward()
        q_value_loss_2.backward()
        self.soft_q_optimizer.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net_1(state, new_action),
                                          self.soft_q_net_2(state, new_action))
        policy_loss = (alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net_1.parameters(), self.soft_q_net_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net_2.parameters(), self.soft_q_net_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        return q_value_loss_1.item(), q_value_loss_2.item(), policy_loss.item()

    def save_model(self, path):
        torch.save(self.modules.state_dict(), path)

    def load_model(self, path):
        self.modules.load_state_dict(torch.load(path))
        self.modules.eval()
