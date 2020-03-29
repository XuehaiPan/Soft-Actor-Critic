import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

from common.buffer import TrajectoryReplayBuffer
from sac.rnn.network import SoftQNetwork, PolicyNetwork, cat_hidden, EncoderWrapper
from sac.trainer import Trainer as OriginalTrainer


class Trainer(OriginalTrainer):
    def __init__(self, env, state_encoder, state_dim, action_dim,
                 hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                 skip_connection, activation,
                 soft_q_lr, policy_lr, alpha_lr, weight_decay,
                 buffer_capacity, device):
        self.env = env
        self.device = device
        self.replay_buffer = TrajectoryReplayBuffer(capacity=buffer_capacity)
        self.n_episodes = 0
        self.episode_steps = []
        self.episode_rewards = []
        self.total_steps = 0

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.training = True

        self.state_encoder = EncoderWrapper(state_encoder, device=device)

        self.soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm,
                                         skip_connection, activation=activation, device=device)
        self.soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                         hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                         activation=activation, device=device)
        self.target_soft_q_net_1 = SoftQNetwork(state_dim, action_dim,
                                                hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                                activation=activation, device=device)
        self.target_soft_q_net_2 = SoftQNetwork(state_dim, action_dim,
                                                hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                                activation=activation, device=device)
        self.target_soft_q_net_1.load_state_dict(self.soft_q_net_1.state_dict())
        self.target_soft_q_net_2.load_state_dict(self.soft_q_net_2.state_dict())

        self.policy_net = PolicyNetwork(state_dim, action_dim,
                                        hidden_dims_before_lstm, hidden_dims_lstm, hidden_dims_after_lstm, skip_connection,
                                        activation=F.relu, device=device)

        self.log_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device), requires_grad=True)

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

        self.soft_q_optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                           self.soft_q_net_1.parameters(),
                                                           self.soft_q_net_2.parameters()),
                                           lr=soft_q_lr, weight_decay=weight_decay)
        self.policy_optimizer = optim.Adam(itertools.chain(self.state_encoder.parameters(),
                                                           self.policy_net.parameters()),
                                           lr=policy_lr, weight_decay=weight_decay)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    def env_sample(self, n_episodes, max_episode_steps, deterministic=False, random_sample=False, render=False, writer=None):
        training = self.training
        self.eval()

        with tqdm.trange(n_episodes * max_episode_steps, desc='Sampling') as pbar:
            for episode in range(n_episodes):
                episode_reward = 0
                episode_steps = 0
                trajectory = []
                hiddens = []
                hidden = self.policy_net.initial_hiddens(batch_size=1)
                state = self.env.reset()
                if render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                for step in range(max_episode_steps):
                    hiddens.append(hidden)

                    if random_sample:
                        action = self.policy_net.random_action()
                    else:
                        encoded_state = self.state_encoder.encode(state)
                        action, hidden = self.policy_net.get_action(encoded_state, hidden, deterministic=deterministic)
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

                    pbar.update()
                    if done:
                        pbar.total -= max_episode_steps - step - 1
                        pbar.refresh()
                        break

                self.replay_buffer.push(*tuple(map(np.stack, zip(*trajectory))), cat_hidden(hiddens, dim=0).detach().cpu())
                pbar.set_postfix({'buffer_size': self.replay_buffer.size})

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

    def update(self, batch_size, step_size=16, normalize_rewards=True, auto_entropy=True, target_entropy=-2.0,
               gamma=0.99, soft_tau=1E-2, epsilon=1E-6):
        self.train()

        # size: (seq_len, batch_size, item_size)
        state, action, reward, next_state, done, hidden = tuple(map(lambda tensor: tensor.to(self.device),
                                                                    self.replay_buffer.sample(batch_size, step_size=step_size)))

        state = self.state_encoder(state)
        with torch.no_grad():
            next_state = self.state_encoder(next_state)

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
        else:
            alpha = 1.0

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
        q_value_loss_1 = self.soft_q_criterion_1(predicted_q_value_1, target_q_value)
        q_value_loss_2 = self.soft_q_criterion_2(predicted_q_value_2, target_q_value)
        q_value_loss = (q_value_loss_1 + q_value_loss_2) / 2.0

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward(retain_graph=True)
        self.soft_q_optimizer.step()

        # Training policy function
        predicted_new_q_value = torch.min(self.soft_q_net_1(state, new_action, hidden)[0],
                                          self.soft_q_net_2(state, new_action, hidden)[0])
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
