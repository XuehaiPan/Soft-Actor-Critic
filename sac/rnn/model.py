from collections import deque
from functools import lru_cache

import numpy as np
import torch

from common.collector import EpisodeCollector
from common.network import cat_hidden
from .network import StateEncoderWrapper
from ..model import Trainer as OriginalTrainer, Tester as OriginalTester


__all__ = ['Trainer', 'Tester']


class Trainer(OriginalTrainer):
    STATE_ENCODER_WRAPPER = StateEncoderWrapper
    COLLECTOR = EpisodeCollector

    def update(self, batch_size, step_size=16,
               normalize_rewards=True, reward_scale=1.0,
               adaptive_entropy=True, target_entropy=-2.0,
               clip_gradient=False, gamma=0.99, soft_tau=0.01, epsilon=1E-6):
        self.train()

        # size: (batch_size * step_size, item_size)
        state, action, reward, next_state, done = self.prepare_batch(batch_size, step_size=step_size)

        return self.update_sac(state, action, reward, next_state, done,
                               normalize_rewards, reward_scale,
                               adaptive_entropy, target_entropy,
                               clip_gradient, gamma, soft_tau, epsilon)

    def prepare_batch(self, batch_size, step_size=16):
        if len(self.episode_cache) < batch_size:
            episodes, lengths = self.replay_buffer.sample(batch_size - len(self.episode_cache),
                                                          min_length=step_size)
            for episode, length in zip(episodes, lengths):
                observation, action, reward, done = episode
                next_observation = np.zeros_like(observation)
                next_observation[:-1] = observation[1:]
                episode = [observation, action, reward, next_observation, done]
                self.episode_cache.append((episode, length, 0,
                                           self.state_encoder.initial_hiddens().cpu()))

        batch = []
        offsets = []
        lengths = []
        episodes = []
        hiddens = []
        for i in range(batch_size):
            episode, length, offset, hidden = self.episode_cache.popleft()
            episodes.append(episode)
            lengths.append(length)
            offsets.append(offset + step_size)
            hiddens.append(hidden)

            # size: (step_size, item_size)
            batch.append([torch.FloatTensor(item[offset:offset + step_size]).to(self.model_device)
                          for item in episode])

        # size: (step_size, batch_size, item_size)
        observation, action, reward, next_observation, done \
            = tuple(map(lambda tensors: torch.stack(tensors, dim=1), zip(*batch)))
        hidden = cat_hidden(hiddens, dim=1).to(self.model_device)

        # size: (step_size, batch_size, item_size)
        state, hidden_last, hidden_all = self.state_encoder(observation, hidden)
        with torch.no_grad():
            # size: (1, batch_size, item_size)
            next_observation_last = next_observation[-1].unsqueeze(dim=0)
            next_state_last, _, _ = self.state_encoder(next_observation_last, hidden_last)

            # size: (step_size, batch_size, item_size)
            next_state = torch.cat([state[:-1].detach(), next_state_last], dim=0)

        for i in reversed(range(batch_size)):
            episode, length, offset = episodes.pop(), lengths.pop(), offsets.pop()
            if offset == length:
                continue

            if offset + step_size <= length:
                hidden = hidden_last[:, i].unsqueeze(dim=0)
            else:  # offset + step_size > length
                hidden = hidden_all[offset + step_size - length - 1, i].unsqueeze(dim=0).unsqueeze(dim=0)
                offset = length - step_size
            self.episode_cache.appendleft((episode, length, offset, hidden.detach().cpu()))

        # size: (batch_size * step_size, item_size)
        state, action, reward, next_state, done \
            = tuple(map(lambda x: x.view(batch_size * step_size, -1),
                        [state, action, reward, next_state, done]))

        return state, action, reward, next_state, done

    @property
    @lru_cache(maxsize=None)
    def episode_cache(self):
        return deque(maxlen=None)


class Tester(OriginalTester):
    STATE_ENCODER_WRAPPER = StateEncoderWrapper
    COLLECTOR = EpisodeCollector
