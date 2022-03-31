from dreamer.rssm import RSSM
from dreamer.configuartion import DreamerConfiguration, RssmConfig
import unittest

import haiku as hk
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


class Fixture:
    def __init__(self):
        c = DreamerConfiguration()
        c.rssm.update(dict(deterministic_size=32, stochastic_size=4))
        c.update(dict(imag_horizon=5))
        self.config = c

        seed = jax.random.PRNGKey(42)

        self.policy = hk.transform(lambda x: tfd.Normal(hk.Linear(1)(x), 1.0))
        seed, subkey = jax.random.split(seed)
        self.policy_params = self.policy.init(subkey, jnp.zeros((1, 36)))
        seed, subkey = jax.random.split(seed)
        self.dummy_observations = jax.random.uniform(subkey, (3, 15, 256))
        seed, subkey = jax.random.split(seed)
        self.dummy_actions = jax.random.uniform(subkey, (3, 14, 1))
        seed, subkey1, subkey2 = jax.random.split(seed, 3)
        self.dummy_state = (jax.random.uniform(subkey1, (3, 4,)),
                            jax.random.uniform(subkey2, (3, 32,)))
        self.seed = seed


class TestRssm(unittest.TestCase):
    def test_call(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
                RSSM(f.config)(prev_state, prev_action, observation))
        f.seed, subkey = jax.random.split(f.seed)
        params = call.init(subkey, tuple(map(lambda x: x[None, 0], f.dummy_state)),
                           f.dummy_actions[None, 0, 0],
                           f.dummy_observations[None, 0, 0])
        (prior, posterior), state = call.apply(
            params, subkey,
            tuple(map(lambda x: x[0], f.dummy_state)),
            f.dummy_actions[0, 0],
            f.dummy_observations[0, 0])
        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, ())
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(state[0].shape, f.dummy_state[0].shape[-1:])
        self.assertEqual(state[1].shape, f.dummy_state[1].shape[-1:])

