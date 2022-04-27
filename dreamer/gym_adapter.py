import gym
import numpy as np
from PIL import Image
from dm_control import suite
from gym import Wrapper, ObservationWrapper
from gym.spaces.box import Box
from gym.wrappers import RescaleAction


def create_env(domain, task, episode_length, action_repeat, seed):
    env = suite.load(domain, task, environment_kwargs=dict(flat_observation=True))
    env = DeepMindSuiteAdapter(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=episode_length)
    render_kwargs = dict(height=64, width=64, camera_id=0)
    env = RepeatAction(env, action_repeat)
    env = RescaleAction(env, -1.0, 1.0)
    env = RenderedObservation(env, (64, 64), render_kwargs)
    env.seed(seed)
    return env


class DeepMindSuiteAdapter(gym.Env):
    def __init__(self, env):
        self._env = env

    def step(self, action):
        time_step = self._env.step(action)
        obs = time_step.observation['observations']
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done, {}

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()['observations']
        return gym.spaces.Box(-np.inf, np.inf, spec.shape, dtype=spec.dtype)

    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs.keys():
            kwargs['camera_id'] = 0
        return self._env.physics.render(**kwargs)

    def reset(self):
        time_step = self._env.reset()
        obs = time_step.observation['observations']
        return obs

    def seed(self, seed=None):
        self._env.task.random.seed(seed)


class RepeatAction(Wrapper):
    def __init__(self, env, repeat):
        super(RepeatAction, self).__init__(env)
        assert repeat > 0, 'repeat must be positve.'
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        repeat = 0
        done = False

        while repeat < self.repeat and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            repeat += 1

        info['repeat'] = repeat
        return obs, total_reward, done, info  # noqa


class RenderedObservation(ObservationWrapper):
    def __init__(self, env, image_size, render_kwargs):
        super(RenderedObservation, self).__init__(env)
        self._size = image_size
        self.observation_space = Box(-0.5, 0.5, image_size + (3,), np.float32)
        self._render_kwargs = render_kwargs

    def observation(self, _):
        image = self.env.render(**self._render_kwargs)
        image = Image.fromarray(image)

        if image.size != self._size:
            image = image.resize(self._size, Image.BILINEAR)

        image = np.array(image, copy=False)
        image = np.clip(image, 0, 255).astype(np.float32)
        image = image / 255.0 - 0.5
        return image
