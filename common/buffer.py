import random
from collections import deque

import numpy as np
import torch
import torch.nn.utils.rnn

from common.network_base import LSTMHidden


class ReplayBuffer(object):
    def __init__(self, capacity=None):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(tuple(args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        # size: (batch_size, item_size)
        # state, action, reward, next_state, done
        return tuple(map(torch.FloatTensor, map(np.stack, zip(*batch))))

    def __len__(self):
        return self.size

    @property
    def capacity(self):
        if self.buffer.maxlen is None:
            return np.inf
        else:
            return self.buffer.maxlen

    @property
    def size(self):
        return len(self.buffer)


class TrajectoryReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=None):
        super().__init__(capacity=capacity)
        self.lengths = deque(maxlen=capacity)

    def push(self, *args):
        super().push(*args)
        self.lengths.append(len(args[0]))

    def sample(self, batch_size, step_size):
        batch = []
        hiddens = []
        for i in range(batch_size):
            while True:
                (state, action, reward, next_state, done, hidden), = random.choices(self.buffer, weights=self.lengths)
                if len(state) >= step_size:
                    offset = random.randint(0, len(state) - step_size)
                    batch.append((state[offset:offset + step_size],
                                  action[offset:offset + step_size],
                                  reward[offset:offset + step_size],
                                  next_state[offset:offset + step_size],
                                  done[offset:offset + step_size]))
                    hiddens.append(hidden[offset].unsqueeze(dim=0))
                    break

        # size: (batch_size, seq_len, item_size)
        # state, action, reward, next_state, done
        batch = map(torch.FloatTensor, map(np.stack, zip(*batch)))

        # size: (seq_len, batch_size, item_size)
        state, action, reward, next_state, done = tuple(map(lambda tensor: tensor.transpose(0, 1), batch))
        hidden = LSTMHidden.cat(hiddens=hiddens, dim=1)
        return state, action, reward, next_state, done, hidden
