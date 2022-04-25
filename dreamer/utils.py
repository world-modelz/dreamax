from typing import Union, List, Dict, Tuple
from contextlib import contextmanager
from threading import Lock
import time

import numpy as np

import haiku as hk


def initializer(name: str) -> hk.initializers.Initializer:
    return {
        'glorot': hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        'he': hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
    }[name]


class Timers:

    def __init__(self, timers: Union[List[str], Tuple[str]]):

        self.timers = {timer: {'time': 0, 'is_timing': False, 'run_times': [], 'lock': Lock()} for timer in timers}

    def start_timer(self, timer: str):
        assert timer in self.timers, f"The timer {timer}, is not a registered timer."

        if self.timers[timer]['is_timing']:
            print(f"The timer {timer}, is already running, so it can't be started.")

        with self.timers[timer]['lock']:
            self.timers[timer]['time'] = time.perf_counter()
            self.timers[timer]['is_timing'] = True

    def stop_timer(self, timer: str):
        assert timer in self.timers, f"The timer {timer}, is not a registered timer."

        if not self.timers[timer]['is_timing']:
            print(f"The timer {timer}, is not running, so it can't be stopped.")

        with self.timers[timer]['lock']:
            self.timers[timer]['run_times'].append(time.perf_counter() - self.timers[timer]['time'])
            self.timers[timer]['time'] = 0
            self.timers[timer]['is_timing'] = False

    @contextmanager
    def timing(self, timer: str):
        self.start_timer(timer)
        yield
        self.stop_timer(timer)

    def collect_times(self) -> Dict[str, float]:

        collect_times = {}

        for timer in self.timers.keys():
            with self.timers[timer]['lock']:
                if len(self.timers[timer]['run_times']) > 0:
                    collect_times[timer] = np.mean(self.timers[timer]['run_times'])
                    self.timers[timer]['run_times'] = []

        return collect_times


class GlobalStepCounter:

    def __init__(self, init_steps: int = 0):
        self._steps = init_steps

        self.lock = Lock()

    def add_steps(self, additional_steps: int):
        assert additional_steps > 0, "'additional_steps', needs to be a positive integer."

        with self.lock:
            self._steps = self._steps + additional_steps

    def add_step(self, add: int = 1):
        with self.lock:
            self._steps = self._steps + add

    @property
    def steps(self) -> int:
        with self.lock:
            steps = self._steps

        return steps