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
        return tuple(map(torch.FloatTensor, map(np.stack, zip(*batch))))

    def __len__(self):
        return len(self.buffer)

    @property
    def capacity(self):
        if self.buffer.maxlen is None:
            return np.inf
        else:
            return self.buffer.maxlen


class TrajectoryReplayBuffer(ReplayBuffer):
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return tuple(map(lambda item_list: list(map(torch.FloatTensor, item_list)), zip(*batch)))
