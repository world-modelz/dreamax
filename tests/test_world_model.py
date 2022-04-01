import unittest

from dreamer.configuartion import DreamerConfiguration
from dreamer.world_model import WorldModel


from .test_rssm import Fixture


def update_config(c: DreamerConfiguration):
    c.update(dict(
        imag_horizon=12,
        rssm=dict(deterministic_size=32, stochastic_size=4),
        encoder=dict(depth=32, kernels=(4, 4, 4, 4)),
        decoder=dict(depth=32, kernels=(5, 5, 6, 6)),
        reward=dict(output_sizes=(400, 400, 400, 400)),
        terminal=dict(output_sizes=(400, 400, 400, 400))
    ))


class TestWorldModel(unittest.TestCase):
    def test_call(self):
        f = Fixture()
