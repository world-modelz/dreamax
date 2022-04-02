import unittest

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp

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
        self.dummy_observations = jax.random.uniform(
            self.rng_split(), (3, 15, 64, 64, 3))
        update_config(self.config)
        self.model = hk.multi_transform(lambda: model(self.config))
        self.params = self.model.init(
            self.seed, self.dummy_observations, self.dummy_actions)


class TestWorldModel(unittest.TestCase):
    def test_call(self):
        f = Fixture2()

        call, *_ = f.model.apply
        (prior, posterior), state = call(
            f.params,
            f.rng_split(),
            tuple(map(lambda x: x[0], f.dummy_state)),
            f.dummy_actions[0, 0],
            f.dummy_observations[0, 0]
        )
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, ())
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(state[0].shape, f.dummy_state[0].shape[-1:])
        self.assertEqual(state[1].shape, f.dummy_state[1].shape[-1:])

    def test_generate(self):
        f = Fixture2()
        _, generate, *_ = f.model.apply
        features, reward, terminal = generate(
            f.params, f.rng_split(),
            jnp.concatenate(f.dummy_state, -1), f.policy, f.policy_params
        )
        self.assertEqual(features.shape, (3, f.config.imag_horizon, 36))
        self.assertEqual(reward.event_shape, ())
        self.assertEqual(tuple(reward.batch_shape), (3, f.config.imag_horizon))
        self.assertEqual(terminal.event_shape, ())
        self.assertEqual(tuple(terminal.batch_shape),
                         (3, f.config.imag_horizon))

    def test_infer(self):
        f = Fixture2()
        _, _, infer, _ = f.model.apply
        (prior, posterior), features, decoded, reward, terminal = infer(
            f.params, f.rng_split(),
            f.dummy_observations, f.dummy_actions
        )
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, (3, 15))
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(features.shape, (3, 15, 36))
        self.assertEqual(reward.event_shape, ())
        self.assertEqual(tuple(reward.batch_shape), (3, 15))
        self.assertEqual(terminal.event_shape, ())
        self.assertEqual(tuple(terminal.batch_shape), (3, 15))
        self.assertEqual(decoded.event_shape, (64, 64, 3))
        self.assertEqual(tuple(decoded.batch_shape), (3, 15))
