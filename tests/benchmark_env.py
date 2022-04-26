# run this with the parent directory as working directory: python -m "tests.benchmark_env"
import time
import logging
import threading
from contextlib import contextmanager

import numpy as np
import gym
from tests.dummy_gym_env import DummyGymEnv
from dreamer.gym_adapter import create_env
from dreamer.utils import Stopwatch


def rollout(env, num_steps):
    obs = env.reset()
    for i in range(num_steps):
        x = env.render(mode='rgb_array')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action=action)
        if done:
            obs = env.reset()


def parallel_rollout(env, num_workers, num_steps):
    worker_threads = []

    for i in range(num_workers):
        w = threading.Thread(target=rollout, args=(env, num_steps), daemon=True)
        worker_threads.append(w)
        w.start()

    for w in worker_threads:
        w.join()


def benchmark_env(name, env, logger):
    logger.info(f'Staring benchmark {name}')

    steps = [100, 200, 400]
    for num_steps in steps:
        sw = Stopwatch()
        with sw.measure():
            rollout(env, num_steps)
        logger.info(f'num_workers: 1; num_steps: {num_steps:>4}; duration: {sw.last:>5.2f}s; total_steps: {num_steps:>5}; rate: {num_steps/sw.last:.2f} steps/s;')

    threads = [2, 4, 8]
    for num_workers in threads:
        for num_steps in steps:
            sw = Stopwatch()
            with sw.measure():
                parallel_rollout(env, num_workers, num_steps)
            total_steps = num_workers * num_steps
            logger.info(f'num_workers: {num_workers}; num_steps: {num_steps:>4}; duration: {sw.last:>5.2f}s; total_steps: {total_steps:>5}; rate: {total_steps/sw.last:.2f} steps/s;')


def make_atari(env_id):
    env = gym.make(env_id)
    return env


def main():
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.INFO)

    # env = DummyGymEnv(observation_space=gym.spaces.Box(-np.inf, np.inf, (64, 64, 3), dtype=np.float32),
    #                       action_space=gym.spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32))
    # benchmark_env('Dummy', env, logger)

    env = make_atari("BreakoutNoFrameskip-v4")
    benchmark_env('BreakoutNoFrameskip-v4', env, logger)

    env = create_env(domain="pendulum", task="swingup", episode_length=1000, action_repeat=1, seed=42)
    benchmark_env('pendulum.swingup', env, logger)


if __name__ == '__main__':
    main()
