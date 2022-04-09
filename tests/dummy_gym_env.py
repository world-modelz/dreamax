import typing
import random
import numpy as np
import gym


class DummyGymEnv(gym.Env):

    def __init__(self, obs_space: typing.Union[gym.Space, typing.List[gym.Space]],
                 action_space: typing.Union[gym.Space, typing.List[gym.Space]]):

        self.obs_space = obs_space
        self.action_space = action_space

    def reset(self):
        if type(self.obs_space) == list:
            obs = [obs.sample() for obs in self.obs_space]
        else:
            obs = self.obs_space.sample()

        return obs

    def step(self, action):

        obs = self.reset()
        reward = random.uniform(0, 1)
        done = np.random.choice([True, False], p=(0.02, 0.98))
        info = {'reward': reward, 'done': done}

        # obs, rewards, done, info
        return obs, reward, done, info
