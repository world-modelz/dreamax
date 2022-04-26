# run this with the parent directory as working directory: python -m "tests.benchmark_env_mp"
import queue
import time
import multiprocessing as mp

import numpy as np

import gym
from dreamer.gym_adapter import create_env
from dreamer.utils import Stopwatch


env = None


def make_atari(env_id):
    env = gym.make(env_id)
    return env


def configure_worker(env_type, **args):
    global env
    if env_type == 'atari':
        env = make_atari(**args)
    elif env_type == 'dm_control':
        env = create_env(**args)
    return 'ok'


def rollout(num_steps):
    obs = env.reset()
    for i in range(num_steps):
        x = env.render(mode='rgb_array')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action=action)
        if done:
            obs = env.reset()
    return 'ok'


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        try:
            result = func(**args)
            output.put(result)
        except Exception as err:
            print(err)


class WorkerProcess:
    def __init__(self, ctx):
        self.input = ctx.Queue()
        self.output = ctx.Queue()
        self.process = ctx.Process(target=worker, args=(self.input, self.output))
        self.process.start()
        self.num_pending = 0

    def enqueue(self, fn, **kwargs):
        self.num_pending += 1
        self.input.put((fn, kwargs))

    def poll(self, block=False, timeout=None):
        response = self.output.get(block, timeout)
        self.num_pending -= 1
        return response

    def configure(self, env_type, **kwargs):
        self.enqueue(configure_worker, env_type=env_type, **kwargs)
        response = self.poll(block=True)
        assert(response == 'ok')

    def stop(self):
        self.input.put('STOP')

    def join(self):
        self.process.join()


def parallel_rollout(ctx, env_config, num_workers, num_steps):
    workers = [WorkerProcess(ctx) for _ in range(num_workers)]

    for w in workers:
        w.configure(**env_config)

    for w in workers:
        w.enqueue(rollout, num_steps=num_steps)

    while any(w.num_pending > 0 for w in workers):
        time.sleep(0.01)
        for w in workers:
            try:
                response = w.poll()
            except queue.Empty:
                pass
            num_pending = sum(w.num_pending for w in workers)

    for w in workers:
        w.stop()

    for w in workers:
        w.join()
        w.process.close()


def benchmark_env(ctx, env):
    steps = [1000, 2000, 4000]
    workers = [2, 4, 8, 16, 32]
    for num_workers in workers:
        for num_steps in steps:
            sw = Stopwatch()
            with sw.measure():
                parallel_rollout(ctx, env, num_workers, num_steps)
            total_steps = num_workers * num_steps
            print(
                f'num_workers: {num_workers}; num_steps: {num_steps:>4}; duration: {sw.last:>5.2f}s; total_steps: {total_steps:>5}; rate: {total_steps/sw.last:.2f} steps/s;')


def main():
    ctx = mp.get_context('spawn')
    #env_config = dict(env_type='atari', env_id="BreakoutNoFrameskip-v4")
    env_config = dict(
        env_type='dm_control',
        domain="pendulum",
        task="swingup",
        episode_length=1000,
        action_repeat=1,
        seed=42)
    benchmark_env(ctx, env_config)


if __name__ == '__main__':
    main()
