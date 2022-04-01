import unittest

import numpy as np
import haiku as hk

from dreamer.configuartion import DreamerConfiguration
from dreamer.world_model import WorldModel

from tests.test_rssm import Fixture


def update_config(c: DreamerConfiguration):
    c.update(dict(
        imag_horizon=12,
        rssm=dict(deterministic_size=32, stochastic_size=4),
        encoder=dict(depth=32, kernels=(4, 4, 4, 4)),
        decoder=dict(depth=32, kernels=(5, 5, 6, 6)),
        reward=dict(output_sizes=(400, 400, 400, 400)),
        terminal=dict(output_sizes=(400, 400, 400, 400))
    ))


def model(config):
    model = WorldModel(np.ones((64, 64, 3)), config)

    def filter_state(prev_state, prev_action, observation):
        return model(prev_state, prev_action, observation)

    def generate_sequence(initial_state, policy, policy_params):
        return model.generate_sequence(initial_state, policy, policy_params)

    def observe_sequence(observations, actions):
        return model.observe_sequence(observations, actions)

    def decode(feature):
        return model.decode(feature)

    def init(observations, actions):
        return model.observe_sequence(observations, actions)

    return init, (filter_state, generate_sequence, observe_sequence, decode)


class Fixture2(Fixture):
    def __init__(self):
        super().__init__()
        update_config(self.config)
        self.model = hk.multi_transform(lambda:model(self.config))
        self.params = self.model.init(
            self.seed, self.dummy_observations, self.dummy_actions)


class TestWorldModel(unittest.TestCase):
    def test_call(self):
        f = Fixture2()

        call, *_ = f.model.apply
        subkey = f.rng_split()
        (prior, posterior), state = call(
            f.params,
            subkey,
            tuple(map(lambda x: x[0], f.dummy_state)),
            f.dummy_actions[0, 0],
            f.dummy_observations[0, 0]
        )
