from dreamer.rssm import RSSM, Posterior, State, Action, Observation
from dreamer.configuartion import DreamerConfiguration
from typing import Optional, Sequence, Tuple
from dreamer.utils import initializer

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class Encoder(hk.Module):
    def __init__(self, depth: int, kernels: Sequence[int], initialization: str):
        super().__init__('Encoder')
        self.depth = depth
        self.kernels = kernels
        self.initialization = initialization

    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        def cnn(x):
            for i, kernel in enumerate(self.kernels):
                depth = 2 ** i * self.depth
                x = jnn.relu(
                    hk.Conv2D(
                        depth,
                        kernel,
                        stride=2,
                        padding='VALID',
                        w_init=initializer(self.initialization)
                    )(x)
                )
            return x

        cnn = hk.BatchApply(cnn)
        return hk.Flatten(2)(cnn(observation))


class Decoder(hk.Module):
    def __init__(
        self,
        depth: int,
        kernels: Sequence[int],
        output_shape: Sequence[int],
        initialization: str
    ):
        super().__init__('Decoder')
        self.depth = depth
        self.kernels = kernels
        self.output_shape = output_shape
        self.initialization = initialization

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        x = hk.BatchApply(
            hk.Linear(32 * self.depth, w_init=initializer(self.initialization))
        )(features)
        x = hk.Reshape((1, 1, 32 * self._depth), 2)(x)

        def transpose_cnn(x):
            for i, kernel in enumerate(self.kernels):
                if i != len(self.kernels) - 1:
                    depth = 2 ** (len(self.kernels) - i - 2) * self.depth
                    x = jnn.relu(
                        hk.Conv2DTranspose(
                            depth,
                            kernel,
                            stride=2,
                            padding='VALID',
                            w_init=initializer(self.initialization)
                        )(x)
                    )
                else:
                    x = hk.Conv2DTranspose(
                        self.output_shape[-1],
                        kernel,
                        stride=2,
                        padding='VALID',
                        w_init=initializer(self.initialization)
                    )(x)
            out = hk.BatchApply(transpose_cnn)(x)
            return tfd.Independent(
                tfd.Normal(out, 1.0), len(self.output_shape)
            )


class DenseDecoder(hk.Module):
    def __init__(self, output_sizes: Sequence[int], dist: str, initialization: str, name: Optional[str] = None):
        super().__init__(name)
        self.output_sizes = output_sizes
        self.initialization = initialization
        self.dist = dict(
            normal=lambda mu: tfd.Normal(mu, 1.0),
            bernoulli=lambda p: tfd.Bernoulli(p)
        )[dist]

    def __call__(self, features: jnp.ndarray):
        mlp = hk.nets.MLP(
            self.output_sizes,
            initializer(self.initialization),
            activation=jnn.elu
        )
        mlp = hk.BatchApply(mlp)
        x = mlp(features)
        x = jnp.squeeze(x, axis=-1)
        return tfd.Independent(self.dist(x), 0)


class WorldModel(hk.Module):
    def __init__(self, observation_space, config: DreamerConfiguration):
        super().__init__()
        self.rssm = RSSM(config)
        self.encoder = Encoder(config.encoder.depth,
                               config.encoder.kernels, config.initialization)
        self.decoder = Decoder(config.decoder.depth, config.decoder.kernels,
                               observation_space.shape, config.initialization)
        self.reward = DenseDecoder(
            config.reward.output_sizes + (1,), 'normal', config.initialization, 'reward')
        self.terminal = DenseDecoder(
            config.terminal.output_sizes + (1,), 'bernoulli', config.initialization, 'terminal')

    def __call__(
        self,
        prev_state: State,
        prev_action: Action,
        observation: Observation
    ) -> Tuple[Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag], State]:
        observation = jnp.squeeze(self.encoder(observation[None, None]))
        return self.rssm(prev_state, prev_action, observation)

    def generate_sequence(
        self,
        initial_features: jnp.ndarray,
        actor: hk.Transformed,
        actor_params: hk.Params,
        actions=None
    ) -> Tuple[jnp.ndarray, tfd.Normal, tfd.Bernoulli]:
        features = self.rssm.generate_sequence(
            initial_features, actor, actor_params, actions)
        reward = self.reward(features)
        terminal = self.terminal(features)
        return features, reward, terminal

    def observe_sequence(
        self,
        observation: Observation,
        action: Action
    ) -> Tuple[
        Tuple[tfd.MultivariateNormalDiag, tfd.MultivariateNormalDiag],
        jnp.ndarray, tfd.Normal, tfd.Normal, tfd.Bernoulli
    ]:
        observation = self.encoder(observation)
        (prior, posterior), features = self.rssm.observe_sequence(observation, action)
        reward = self.reward(features)
        terminal = self.terminal(features)
        decoded = self.decode(features)
        return (prior, posterior), features, decoded, reward, terminal

    def decode(self, features: jnp.ndarray) -> tfd.Normal:
        return self.decoder(features)


class Actor(hk.Module):
    def __init__(self, output_sizes: Sequence[int], min_stddev: float, initialization: str):
        super().__init__()
        self.output_sizes = output_sizes
        self.min_stddev = min_stddev
        self.initialization = initialization

    def __call__(self, observation):
        mlp = hk.nets.MLP(self.output_sizes, activation=jnn.elu,
                          w_init=initializer(self.initialization))
        mu, stddev = jnp.split(mlp(observation), 2, axis=-1)
        init_std = np.log(np.exp(5.0) - 1.0).astype(stddev.dtype)
        stddev = jnn.softplus(stddev + init_std) + self.min_stddev
        multivariate_normal_diag = tfd.Normal(5.0 * jnn.tanh(mu / 5.0), stddev)
        # Squash actions to [-1, 1]
        squashed = tfd.TransformedDistribution(
            multivariate_normal_diag, StableTanhBijector())
        dist = tfd.Independent(squashed, 1)
        return SampleDist(dist)


class StableTanhBijector(tfb.Tanh):
    def __init__(self, validate_args=False, name='tanh_stable_bijector'):
        super().__init__(validate_args=validate_args,
                         name=name)

    def _inverse(self, y):
        dtype = y.dtype
        y = y.astype(jnp.float32)
        y = jnp.clip(y, -0.99999997, 0.99999997)
        y = jnp.arctanh(y)
        return y.astype(dtype)


class SampleDist:
    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self, seed):
        samples = self._dist.sample(self._samples, seed=seed)
        return jnp.mean(samples, 0)

    def mode(self, seed):
        sample = self._dist.sample(self._samples, seed=seed)
        logprob = self._dist.log_prob(sample)
        return sample[jnp.argmax(logprob, 0).squeeze()]

    def entropy(self, seed):
        sample = self._dist.sample(self._samples, seed=seed)
        logprob = self.log_prob(sample)
        return -jnp.mean(logprob, 0)
