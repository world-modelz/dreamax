from dreamer.utils import initializer
from dreamer.configuration import DreamerConfiguration, RssmConfig
from distutils.command.config import config
from typing import Tuple, Optional

import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

State = Tuple[jnp.ndarray, jnp.ndarray]
Action = jnp.ndarray
obs = jnp.ndarray


class ZHeadLayer(hk.Module):
    def __init__(self, config: RssmConfig, initialization: str):
        super().__init__()
        self.config = config
        self.initialization = initialization

    def __call__(self, prev_state: State, prev_action: Action) -> Tuple[tfd.MultivariateNormalDiag, State]:

        # Unstack state
        stoch, det = prev_state
        cat = jnp.concatenate([prev_action, stoch], -1)

        x = hk.Linear(self.config.deterministic_size, name='h1', w_init=initializer(self.initialization))(cat)
        x = jnn.elu(x)

        x, det = hk.GRU(self.config.deterministic_size, w_i_init=initializer(
            self.initialization), w_h_init=hk.initializers.Orthogonal())(x, det)

        x = hk.Linear(self.config.hidden, name='h2', w_init=initializer(self.initialization))(x)
        x = jnn.elu(x)

        # Create Dist
        x = hk.Linear(self.config.stochastic_size * 2, name='h3', w_init=initializer(self.initialization))(x)
        mean, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        z_head_dist = tfd.MultivariateNormalDiag(mean, stddev)

        # Sample from Dist
        sample = z_head_dist.sample(seed=hk.next_rng_key())

        # Stack state
        state = (sample, det)

        return z_head_dist, state


class ZLayer(hk.Module):
    def __init__(self, config: RssmConfig, inititialization: str):
        super().__init__()

        self.config = config
        self.initialization = inititialization

    def __call__(self, prev_state: State, obs: obs) -> Tuple[tfd.MultivariateNormalDiag, State]:

        # Unstack state
        _, det = prev_state
        cat = jnp.concatenate([det, obs], -1)

        x = hk.Linear(self.config.hidden, name='h1', w_init=initializer(self.initialization))(cat)
        x = jnn.elu(x)

        # Create Dist
        x = hk.Linear(self.config.stochastic_size * 2, name='h2', w_init=initializer(self.initialization))(x)
        mean, stddev = jnp.split(x, 2, -1)
        stddev = jnn.softplus(stddev) + 0.1
        z_dist = tfd.MultivariateNormalDiag(mean, stddev)

        # Sample from Dist
        z_sample = z_dist.sample(seed=hk.next_rng_key())

        # Stack state
        state = (z_sample, det)

        return z_dist, state


def init_state(batch_size: int, stochastic_size: int, deterministic_size: int,
               dtype: Optional[jnp.dtype] = jnp.float32) -> State:
    return (jnp.zeros((batch_size, stochastic_size), dtype), jnp.zeros((batch_size, deterministic_size), dtype))


class RSSM(hk.Module):
    def __init__(self, config: DreamerConfiguration):
        super().__init__()

        self.config = config
        self.z_head = ZHeadLayer(config.rssm, config.initialization)
        self.z = ZLayer(config.rssm, config.initialization)

    def __call__(self, prev_state: State, prev_action: Action, obs: obs) \
            -> Tuple[Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag], State]:

        z_head_dist, state = self.z_head(prev_state, prev_action)
        z_dist, state = self.z(state, obs)

        return (z_head_dist, z_dist), state

    def generate_sequence(
            self,
            initial_features: jnp.ndarray,
            actor: hk.Transformed,
            actor_params: hk.Params,
            actions=None) -> jnp.ndarray:

        horizon = self.config.imag_horizon if actions is None else actions.shape[1]
        sequence = jnp.zeros(
            (initial_features.shape[0],
             horizon,
             self.config.rssm.stochastic_size +
             self.config.rssm.deterministic_size))
        state = jnp.split(initial_features, (self.config.rssm.stochastic_size,), -1)
        keys = hk.next_rng_keys(horizon)

        for t, key in enumerate(keys):

            if actions is None:
                state_stack = jnp.concatenate(state, -1)
                state_stack_stop = jax.lax.stop_gradient(state_stack)
                action_dist = actor.apply(actor_params, key, state_stack_stop)
                action = action_dist.sample(seed=key)
            else:
                action = actions[:, t]

            _, state = self.z_head(state, action)
            state_stack = jnp.concatenate(state, -1)
            sequence = sequence.at[:, t].set(state_stack)

        return sequence

    def observe_sequence(self, obss: obs, actions: Action) \
            -> Tuple[Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag], jnp.ndarray]:

        z_heads, z_s = [], []
        features = jnp.zeros(obss.shape[:2] +
                             (self.config.rssm.stochastic_size +
                              self.config.rssm.deterministic_size,))
        state = init_state(obss.shape[0], self.config.rssm.stochastic_size, self.config.rssm.deterministic_size)

        # Unroll over sequence
        for t in range(obss.shape[1]):
            (z_head_dist, z_dist), state = self.__call__(state, actions[:, t], obss[:, t])

            # Add mean and stddev to buffer list.
            z_heads.append((z_head_dist.mean(), z_head_dist.stddev()))
            z_s.append((z_dist.mean(), z_dist.stddev()))

            # Set features at time step t
            features = features.at[:, t].set(jnp.concatenate(state, -1))

        def joint_mvn(dists):
            mus, stddevs = jnp.asarray(list(zip(*dists))).transpose((0, 2, 1, 3))
            return tfd.MultivariateNormalDiag(mus, stddevs)

        # Stack unrolled mean and stddev
        z_head_stack = joint_mvn(z_heads)
        z_stack = joint_mvn(z_s)

        return (z_head_stack, z_stack), features
