from collections import deque

import gym
import numpy as np
import torchvision.transforms as transforms
from gym.spaces import Box


__all__ = [
    'FlattenedAction', 'NormalizedAction',
    'FlattenedObservation', 'VisionObservation', 'ConcatenatedObservation'
]

try:
    import pybullet_envs
except ImportError:
    pass


class FlattenedAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.action_space = Box(low=self.env.action_space.low.ravel(),
                                high=self.env.action_space.high.ravel(),
                                dtype=np.float32)

    def action(self, action):
        return np.ravel(action)

    def reverse_action(self, action):
        return np.reshape(action, self.env.action_space.shape)


class NormalizedAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.action_space = Box(low=-1.0,
                                high=1.0,
                                shape=self.env.action_space.shape,
                                dtype=np.float32)

    def action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high

        action = low + 0.5 * (action + 1.0) * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.env.action_space.low
        high = self.env.action_space.high

        action = np.clip(action, low, high)
        action = 2.0 * (action - low) / (high - low) - 1.0

        return action


class FlattenedObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env=env)
        self.observation_space = Box(low=self.env.observation_space.low.ravel(),
                                     high=self.env.observation_space.high.ravel(),
                                     dtype=np.float32)

    def observation(self, observation):
        return np.ravel(observation)


class VisionObservation(gym.ObservationWrapper):
    def __init__(self, env, image_size=(128, 128)):
        super().__init__(env=env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(3, *image_size), dtype=np.float32)
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=image_size),
            transforms.ToTensor()
        ])

        self.unwrapped_observation_space = self.env.observation_space
        self.unwrapped_observation = None

    def observation(self, observation):
        self.unwrapped_observation = observation

        obs = self.render(mode='rgb_array')
        obs = self.transform(obs)

        return obs


class ConcatenatedObservation(gym.ObservationWrapper):
    def __init__(self, env, n_frames=3, dim=0):
        super().__init__(env=env)

        self.observation_space = Box(low=np.concatenate([self.env.observation_space.low] * n_frames, axis=dim),
                                     high=np.concatenate([self.env.observation_space.high] * n_frames, axis=dim),
                                     dtype=self.env.observation_space.dtype)

        self.queue = deque(maxlen=n_frames)
        self.dim = dim

    def reset(self, **kwargs):
        self.queue.clear()
        return super().reset(**kwargs)

    def observation(self, observation):
        while len(self.queue) < self.n_frames - 1:
            self.queue.append(observation)
        self.queue.append(observation)
        return np.concatenate(self.queue, axis=self.dim)

    @property
    def n_frames(self):
        return self.queue.maxlen
