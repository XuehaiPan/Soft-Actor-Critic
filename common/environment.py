import gym
import numpy as np
from gym.spaces import Box


__all__ = ['FlattenedAction', 'NormalizedAction', 'FlattenedObservation']


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
