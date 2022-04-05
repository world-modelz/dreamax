import unittest
import numpy as np
import gym
import jax

from tests.dummy_gym_env import DummyGymEnv

from dreamer.replay_buffer import ReplayBuffer
from dreamer.configuration import ReplayBufferConfig


def interact(env, episodes, episode_length, buffer):
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    for _ in range(episodes):
        observation = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            terminal = done and not info.get('TimeLimit.truncated', False)
            buffer.store(dict(observation=observation,
                              next_observation=next_observation,
                              action=action.astype(np.float32),
                              reward=np.array(reward, np.float32),
                              terminal=np.array(terminal, np.bool_),
                              info=info))
            observation = next_observation


class TestReplayBuffer(unittest.TestCase):
    def test_store(self):
        env = DummyGymEnv(observation_space=gym.spaces.Box(-np.inf, np.inf, (64, 64, 3), dtype=np.float32),
                          action_space=gym.spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32))

        episode_length = 10
        episodes = 3

        buffer_config = dict(capacity=5, batch_size=2, sequence_length=4)
        buffer_config = ReplayBufferConfig(buffer_config)

        buffer = ReplayBuffer(
            config=buffer_config,
            observation_space=env.observation_space,
            action_space=env.action_space,
            precision=16,
            seed=0)
        interact(env, episodes, episode_length, buffer)
        self.assertEqual(buffer.idx + 1, episodes)

    def test_sample(self):
        with jax.disable_jit():

            env = DummyGymEnv(observation_space=gym.spaces.Box(-np.inf, np.inf, (3, ), dtype=np.float32),
                              action_space=gym.spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32))

            episode_length = 10
            episodes = 3

            buffer_config = dict(capacity=5, batch_size=2, sequence_length=4)
            buffer_config = ReplayBufferConfig(buffer_config)

            buffer = ReplayBuffer(
                config=buffer_config,
                observation_space=env.observation_space,
                action_space=env.action_space,
                precision=16,
                seed=0)
            interact(env, episodes, episode_length, buffer)
            samples = list(buffer.sample(2))
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[0]['observation'].shape, (2, 4, 3))
