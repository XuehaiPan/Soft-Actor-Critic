import numpy as np
import torch.multiprocessing as mp


__all__ = ['ReplayBuffer', 'EpisodeReplayBuffer']


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
        return tuple(map(np.stack, zip(*batch)))

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


class EpisodeReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, initializer, Value=mp.Value, Lock=mp.Lock()):
        super().__init__(capacity=capacity, initializer=initializer, Value=Value, Lock=Lock)
        self.lengths = initializer()
        self.buffer_size = Value('L', 0)
        self.n_total_episodes = Value('L', 0)
        self.length_mean = Value('f', 0.0)
        self.length_square_mean = Value('f', 0.0)

    def push(self, *args):
        items = tuple(args)
        length = len(args[0])
        with self.lock:
            buffer_len = len(self.buffer)
            if self.size + length <= self.capacity:
                self.buffer.append(items)
                self.lengths.append(length)
                self.buffer_size.value += length
                self.offset = buffer_len + 1
            else:
                self.offset %= buffer_len
                self.buffer[self.offset] = items
                self.buffer_size.value += length - self.lengths[self.offset]
                self.lengths[self.offset] = length
                self.offset = (self.offset + 1) % buffer_len
            self.n_total_episodes.value += 1
            self.length_mean.value += (length - self.length_mean.value) \
                                      / self.n_total_episodes.value
            self.length_square_mean.value += (length * length - self.length_square_mean.value) \
                                             / self.n_total_episodes.value

    def sample(self, batch_size, min_length=16):
        length_mean = self.length_mean.value
        length_square_mean = self.length_square_mean.value
        length_stddev = np.sqrt(length_square_mean - length_mean * length_mean)

        if length_stddev / length_mean < 0.1:
            weights = np.ones(shape=(len(self.lengths),))
        else:
            weights = np.asanyarray(list(self.lengths))
        weights = weights / weights.sum()

        episodes = []
        lengths = []
        for i in range(batch_size):
            while True:
                index = np.random.choice(len(weights), p=weights)

                # size: (length, item_size)
                # observation, action, reward, done
                items = self.buffer[index]
                length = len(items[0])
                if length >= min_length:
                    episodes.append(items)
                    lengths.append(length)
                    break

        return episodes, lengths

    @property
    def size(self):
        return self.buffer_size.value
