from mimetypes import init
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

        self.seed = jax.random.PRNGKey(42)

        self.policy = hk.transform(lambda x: tfd.Normal(hk.Linear(1)(x), 1.0))
        self.policy_params = self.policy.init(
            self.rng_split(), jnp.zeros((1, 36)))
        self.dummy_observations = jax.random.uniform(
            self.rng_split(), (3, 15, 256))
        self.dummy_actions = jax.random.uniform(self.rng_split(), (3, 14, 1))
        subkey1, subkey2 = self.rng_split(3)
        self.dummy_state = (jax.random.uniform(subkey1, (3, 4,)),
                            jax.random.uniform(subkey2, (3, 32,)))

    def rng_split(self, num=2):
        self.seed, *remain = jax.random.split(self.seed, num)
        return remain[0] if len(remain) == 1 else remain


class TestRssm(unittest.TestCase):
    def test_call(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
                RSSM(f.config)(prev_state, prev_action, observation)
        )
        subkey = f.rng_split()
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

    def test_generate(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
                RSSM(f.config)(prev_state, prev_action, observation)
        )
        generate = hk.transform(
            lambda initial_state, policy, policy_params:
                RSSM(f.config).generate_sequence(
                    initial_state, policy, policy_params
                )
        )
        subkey = f.rng_split()
        params = call.init(
            subkey,
            tuple(map(lambda x: x[None, 0], f.dummy_state)),
            f.dummy_actions[None, 0, 0],
            f.dummy_observations[None, 0, 0]
        )
        output = generate.apply(
            params,
            subkey,
            jnp.concatenate(f.dummy_state, -1),
            f.policy,
            f.policy_params
        )
        self.assertEqual(output.shape, (3, 5, 36))

    def test_infer(self):
        f = Fixture()
        call = hk.transform(
            lambda prev_state, prev_action, observation:
                RSSM(f.config)(prev_state, prev_action, observation)
        )
        infer = hk.transform(
            lambda observations, actions:
                RSSM(f.config).observe_sequence(observations, actions)
        )
        subkey = f.rng_split()
        params_infer = infer.init(
            subkey, f.dummy_observations, f.dummy_actions)
        outputs_infer = infer.apply(
            params_infer,
            subkey,
            f.dummy_observations,
            f.dummy_actions
        )
        (prior, posterior), outs = outputs_infer

        self.assertEqual(prior.event_shape, (4,))
        self.assertEqual(prior.batch_shape, (3, 15))
        self.assertEqual(prior.event_shape, posterior.event_shape)
        self.assertEqual(prior.batch_shape, posterior.batch_shape)
        self.assertEqual(outs.shape, (3, 15, 36))

        params_call = call.init(
            subkey,
            tuple(map(lambda x: x[None, 0], f.dummy_state)),
            f.dummy_actions[None, 0, 0],
            f.dummy_observations[None, 0, 0]
        )
        *_, outs_call = infer.apply(
            params_call,
            subkey,
            f.dummy_observations,
            f.dummy_actions
        )
        self.assertTrue(jnp.equal(outs, outs_call).all())
