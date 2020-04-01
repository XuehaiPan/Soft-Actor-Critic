import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from common.buffer import ReplayBuffer
from sac.network import SoftQNetwork, PolicyNetwork, EncoderWrapper


__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, env, state_encoder, state_dim, action_dim, hidden_dims, activation,
                 initial_alpha, soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 buffer_capacity, device):
        self.env = env
        self.device = device
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.n_episodes = 0
        self.episode_steps = []
        self.episode_rewards = []
        self.total_steps = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = EncoderWrapper(state_encoder, device=device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=device)
        self.target_soft_q_net_1 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=device)
        self.target_soft_q_net_2 = SoftQNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=device)
        self.target_soft_q_net_1.load_state_dict(self.soft_q_net_1.state_dict())
        self.target_soft_q_net_2.load_state_dict(self.soft_q_net_2.state_dict())
        self.target_soft_q_net_1.eval()
        self.target_soft_q_net_2.eval()

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dims, activation=activation, device=device)

        self.log_alpha = nn.Parameter(torch.tensor([[np.log(initial_alpha)]], dtype=torch.float32, device=device), requires_grad=True)

        self.modules = nn.ModuleDict({
            'state_encoder': self.state_encoder,
            'soft_q_net_1': self.soft_q_net_1,
            'soft_q_net_2': self.soft_q_net_2,
            'target_soft_q_net_1': self.target_soft_q_net_1,
            'target_soft_q_net_2': self.target_soft_q_net_2,
            'policy_net': self.policy_net,
            'params': nn.ParameterDict({'log_alpha': self.log_alpha})
        })

        self.soft_q_criterion_1 = nn.MSELoss()
        self.soft_q_criterion_2 = nn.MSELoss()

        self.optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                    self.soft_q_net_1.parameters(),
                                                    self.soft_q_net_2.parameters(),
                                                    self.policy_net.parameters()),
                                    lr=soft_q_lr, weight_decay=weight_decay)
        self.policy_loss_weight = policy_lr / soft_q_lr
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def print_info(self):
        print(f'env = {self.env}')
        print(f'state_dim = {self.state_dim}')
        print(f'action_dim = {self.action_dim}')
        print(f'device = {self.device}')
        print(f'buffer_capacity = {self.replay_buffer.capacity}')
        print('Modules:', self.modules)

    def env_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False, writer=None):
        training = self.training
        self.eval()

        with tqdm.trange(n_episodes * max_episode_steps, desc='Sampling') as pbar:
            for episode in range(n_episodes):
                episode_reward = 0
                episode_steps = 0
                observation = self.env.reset()
                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                for step in range(max_episode_steps):
                    if random_sample:
                        action = self.env.action_space.sample()
                    else:
                        state = self.state_encoder.encode(observation)
                        action = self.policy_net.get_action(state, deterministic=deterministic)
                    next_observation, reward, done, _ = self.env.step(action)
                    if render:
                        try:
                            self.env.render()
                        except Exception:
                            pass

                    episode_reward += reward
                    episode_steps += 1
                    self.replay_buffer.push(observation, action, [reward], next_observation, [done])
                    observation = next_observation

                    pbar.set_postfix({'buffer_size': self.replay_buffer.size})
                    pbar.update()
                    if done:
                        pbar.total -= max_episode_steps - step - 1
                        pbar.refresh()
                        break

                if not random_sample:
                    self.n_episodes += 1
                    self.episode_steps.append(episode_steps)
                    self.episode_rewards.append(episode_reward)
                    self.total_steps += episode_steps
                    average_reward = episode_reward / episode_steps
                    if writer is not None:
                        writer.add_scalar(tag='sample/cumulative_reward', scalar_value=episode_reward, global_step=self.n_episodes)
                        writer.add_scalar(tag='sample/average_reward', scalar_value=average_reward, global_step=self.n_episodes)
                        writer.add_scalar(tag='sample/episode_steps', scalar_value=episode_steps, global_step=self.n_episodes)
        if not random_sample and writer is not None:
            writer.flush()

        self.train(mode=training)

    def update(self, batch_size, normalize_rewards=True, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        self.train()

        # size: (batch_size, item_size)
        observation, action, reward, next_observation, done = tuple(map(lambda tensor: tensor.to(self.device),
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
        for target_param, param in zip(self.target_soft_q_net_1.parameters(), self.soft_q_net_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        for target_param, param in zip(self.target_soft_q_net_2.parameters(), self.soft_q_net_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
        return soft_q_loss.item(), policy_loss.item(), alpha.item()

    def save_model(self, path):
        torch.save(self.modules.state_dict(), path)

    def load_model(self, path):
        self.modules.load_state_dict(torch.load(path, map_location=self.device))
        self.target_soft_q_net_1.eval()
        self.target_soft_q_net_2.eval()

    def train(self, mode=True):
        self.training = mode
        self.modules.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)
