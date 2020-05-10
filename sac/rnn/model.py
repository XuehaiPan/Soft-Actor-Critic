from collections import deque

import numpy as np
import torch

from common.collector import EpisodeCollector
from common.network import cat_hidden
from sac.model import ModelBase, TrainerBase
from sac.rnn.network import StateEncoderWrapper


__all__ = ['Trainer', 'Tester']


class Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        kwargs.update(state_encoder_wrapper=StateEncoderWrapper,
                      collector=EpisodeCollector)
        super().__init__(*args, **kwargs)
        self.episode_cache = deque(maxlen=None)

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
                self.episode_cache.append((episode, length, 0, self.state_encoder.initial_hiddens()))

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
        hidden = cat_hidden(hiddens, dim=1)

        # size: (step_size, batch_size, item_size)
        state, hidden_last, hidden_all = self.state_encoder(observation, hidden)
        with torch.no_grad():
            next_hidden = hidden_all[0].unsqueeze(dim=0)

            # size: (step_size, batch_size, item_size)
            next_state, _, _ = self.state_encoder(next_observation, next_hidden)

        for i in reversed(range(batch_size)):
            episode, length, offset = tuple(map(list.pop, [episodes, lengths, offsets]))
            if offset == length:
                continue

            if offset + step_size <= length:
                hidden = hidden_last[:, i].unsqueeze(dim=0)
            else:  # offset + step_size > length
                hidden = hidden_all[offset + step_size - length - 1, i].unsqueeze(dim=0).unsqueeze(dim=0)
                offset = length - step_size
            self.episode_cache.appendleft((episode, length, offset, hidden.detach()))

        # size: (step_size * batch_size, item_size)
        state, action, reward, next_state, done \
            = tuple(map(lambda x: x.view(batch_size * step_size, -1),
                        [state, action, reward, next_state, done]))

        return state, action, reward, next_state, done


class Tester(ModelBase):
    def __init__(self, *args, **kwargs):
        kwargs.update(state_encoder_wrapper=StateEncoderWrapper,
                      collector=EpisodeCollector)
        super().__init__(*args, **kwargs)

        self.eval()
