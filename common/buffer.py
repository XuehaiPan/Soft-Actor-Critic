import random
from collections import deque

import numpy as np
import torch


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
        return len(self.buffer)

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
    def sample(self, batch_size, enforce_sorted=True):
        batch = random.sample(self.buffer, batch_size)

        if enforce_sorted:
            batch.sort(key=lambda trajectory: len(trajectory[0]), reverse=True)

        # size: (batch_size, seq_len, item_size)
        # batch_trajectory_{state, action, reward, next_state, done}
        return tuple(map(lambda item_list: list(map(torch.FloatTensor, item_list)), zip(*batch)))
