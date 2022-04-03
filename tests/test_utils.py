import unittest
import time

from dreamer.utils import Timers


class TestTimers(unittest.TestCase):

    def test_timer(self):

        timers = Timers(['timer_a', 'timer_b', 'timer_c'])

        for _ in range(2):

            with timers.timing('timer_a'):
                for __ in range(5):
                    with timers.timing('timer_b'):
                        time.sleep(0.5)

        collect_times = timers.collect_times()
        self.assertNotIn('timer_c', collect_times)

        self.assertAlmostEqual(collect_times['timer_a'], 2.5, 2)
        self.assertAlmostEqual(collect_times['timer_b'], 0.5, 2)
