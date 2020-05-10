import numpy as np
import torch
import torch.multiprocessing as mp

from common.network import cat_hidden


__all__ = ['ReplayBuffer', 'TrajectoryReplayBuffer']


class ReplayBuffer(object):
    def __init__(self, capacity, initializer, Value=mp.Value, Lock=mp.Lock):
        self.capacity = capacity
        self.buffer = initializer()
        self.buffer_offset = Value('L', 0)
        self.lock = Lock()

    def push(self, *args):
        items = tuple(args)
        with self.lock:
            if self.size < self.capacity:
                self.buffer.append(items)
            else:
                self.buffer[self.offset] = items
            self.offset = (self.offset + 1) % self.capacity

    def extend(self, trajectory):
        with self.lock:
            for items in map(tuple, trajectory):
                if self.size < self.capacity:
                    self.buffer.append(items)
                else:
                    self.buffer[self.offset] = items
                self.offset = (self.offset + 1) % self.capacity

    def sample(self, batch_size):
        batch = []
        for i in np.random.randint(self.size, size=batch_size):
            batch.append(self.buffer[i])

        # size: (batch_size, item_size)
        # observation, action, reward, next_observation, done
        return tuple(map(torch.FloatTensor, map(np.stack, zip(*batch))))

    def __len__(self):
        return self.size

    @property
    def size(self):
        return len(self.buffer)

    @property
    def offset(self):
        return self.buffer_offset.value

    @offset.setter
    def offset(self, value):
        self.buffer_offset.value = value


class TrajectoryReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, initializer, Value=mp.Value, Lock=mp.Lock()):
        super().__init__(capacity=capacity, initializer=initializer, Value=Value, Lock=Lock)
        self.lengths = initializer()
        self.buffer_size = Value('L', 0)

    def push(self, *args):
        items = tuple(args)
        length = len(args[0])
        with self.lock:
            if self.size + length <= self.capacity:
                self.buffer.append(items)
                self.lengths.append(length)
                self.buffer_size.value += length
                self.offset += 1
            else:
                self.buffer[self.offset] = items
                self.buffer_size.value += length - self.lengths[self.offset]
                self.lengths[self.offset] = length
                self.offset = (self.offset + 1) % (len(self.buffer))

    def sample(self, batch_size, step_size):
        batch = []
        hiddens = []
        with self.lock:
            lengths = np.asanyarray(self.lengths)
        weights = lengths / lengths.sum()
        while len(batch) < batch_size:
            while True:
                index = np.random.choice(len(weights), p=weights)
                *items, hidden = self.buffer[index]
                length = len(items[0])
                if length < step_size:
                    continue
                offsets = np.arange(step_size, length - step_size, step_size, dtype=np.int64)
                np.random.shuffle(offsets)
                for offset in [length - step_size, 0, *offsets]:
                    batch.append([item[offset:offset + step_size] for item in items])
                    hiddens.append(hidden[offset].float().unsqueeze(dim=0))
                if len(batch) >= batch_size:
                    break
        while len(batch) > batch_size:
            batch.pop()
            hiddens.pop()

        # size: (batch_size, seq_len, item_size)
        # observation, action, reward, next_observation, done
        batch = map(torch.FloatTensor, map(np.stack, zip(*batch)))

        # size: (seq_len, batch_size, item_size)
        batch = tuple(map(lambda tensor: tensor.transpose(0, 1), batch))
        hidden = cat_hidden(hiddens=hiddens, dim=1)
        return (*batch, hidden)

    @property
    def size(self):
        return self.buffer_size.value
