import unittest
import numpy as np
from dreamer.gym_adapter import create_env


class TestGymAdapter(unittest.TestCase):
    def test_create_pendulum_swingup_env(self):
        env = create_env(
            domain="pendulum",
            task="swingup",
            episode_length=1000,
            action_repeat=2,
            seed=42,
        )

        # basic render function
        x = env.render()
        self.assertEqual(x.shape, (240, 320, 3))
        self.assertEqual(x.dtype, np.uint8)

        # test shape and dtype of action and observation spaces
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.action_space.dtype, np.float32)
        self.assertEqual(env.observation_space.shape, (64, 64, 3))
        self.assertEqual(env.observation_space.dtype, np.float32)

        # reset env
        obs = env.reset()
        self.assertEqual(obs.shape, env.observation_space.shape)

        # execute random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action=action)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertFalse(done)
