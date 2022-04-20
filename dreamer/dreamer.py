from typing import Mapping, Tuple
import functools
from collections import defaultdict
import os
import pickle

import gym
import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

from dreamer.configuration import DreamerConfiguration
from dreamer.replay_buffer import ReplayBuffer
from dreamer.rssm import obs, init_state
from dreamer.logger import TrainingLogger


PRNGKey = jnp.ndarray
State = Tuple[jnp.ndarray, jnp.ndarray]
Action = jnp.ndarray
obs = np.ndarray
Batch = Mapping[str, np.ndarray]
tfd = tfp.distributions
LearningState = Tuple[hk.Params, optax.OptState]


def get_mixed_precision_policy(precision):
    policy = (
        "params=float32,compute=float"
        + str(precision)
        + ",output=float"
        + str(precision)
    )
    return jmp.get_policy(policy)


def discount_(factor, length):
    d = np.cumprod(factor * np.ones((length - 1,)))
    d = np.concatenate([np.ones((1,)), d])
    return d


class Dreamer:
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        model: hk.MultiTransformed,
        actor: hk.Transformed,
        critic: hk.Transformed,
        logger: TrainingLogger,
        config: DreamerConfiguration,
        precision=get_mixed_precision_policy(16),
        is_training_instance: bool = True,
    ):

        self.action_space = action_space
        self.config = config
        self.rng_seq = hk.PRNGSequence(config.seed)
        self.precision = precision
        self.dtype = precision.compute_dtype
        self.is_training_instance = is_training_instance

        self.params = {}
        self.opt_state = {}

        features_example = jnp.concatenate(self.init_state, -1)[None, None].astype(self.dtype)
        env_space_example = (obs_space.sample()[None, None].astype(self.dtype),
                             action_space.sample()[None, None].astype(self.dtype))

        self.model = model
        self.actor = actor
        self.critic = critic

        self.params['model'] = self.model.init(next(self.rng_seq), *env_space_example)
        self.params['actor'] = self.actor.init(next(self.rng_seq), features_example)

        if self.is_training_instance:
            self.params['critic'] = self.critic.init(next(self.rng_seq), features_example)

            self.model_optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.model_opt.clip),
                optax.scale_by_adam(eps=self.config.model_opt.eps),
                optax.scale(-self.config.model_opt.lr),
            )

            self.actor_optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.actor_opt.clip),
                optax.scale_by_adam(eps=self.config.actor_opt.eps),
                optax.scale(-self.config.actor_opt.lr),
            )

            self.critic_optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.actor_opt.clip),
                optax.scale_by_adam(eps=self.config.actor_opt.eps),
                optax.scale(-self.config.actor_opt.lr),
            )

            self.opt_state['model'] = self.model_optimizer.init(self.params['model'])
            self.opt_state['actor'] = self.actor_optimizer.init(self.params['actor'])
            self.opt_state['critic'] = self.actor_optimizer.init(self.params['critic'])

        self.logger = logger
        self.training_step = 0

    @functools.partial(jax.jit, static_argnums=(0, 7))
    def policy(
        self,
        prev_state: State,
        prev_action: Action,
        obs: np.ndarray,
        key: PRNGKey,
        training=True
    ):
        filter_, *_ = self.model.apply
        key, subkey = jax.random.split(key)
        obs = obs.astype(self.precision.compute_dtype)
        _, current_state = filter_(self.params['model'], key, prev_state, prev_action, obs)
        features = jnp.concatenate(current_state, -1)[None]
        policy = self.actor.apply(self.params['actor'], features)
        action = policy.sample(seed=key) if training else policy.mode(seed=key)
        action = jnp.squeeze(action, axis=0)

        return jnp.clip(action.astype(jnp.float32), -1, 1), current_state, action

    def get_inital_state(self):
        return self.init_state, jnp.zeros(self.action_space.shape, self.dtype)

    @property
    def init_state(self):
        state = init_state(1, self.config.rssm.stochastic_size, self.config.rssm.deterministic_size,
                           self.precision.compute_dtype)
        return jax.tree_map(lambda x: x.squeeze(0), state)

    @functools.partial(jax.jit, static_argnums=0)
    def update(
        self,
        batch: Batch,
        key: PRNGKey
    ) -> dict:

        assert self.is_training_instance, "Update function can only be called with training instances."

        key, subkey = jax.random.split(key)
        model_report, features = self.update_model(batch, subkey)

        key, subkey = jax.random.split(key)
        actor_report, (generated_features, lambda_values) = self.update_actor(features,  subkey)
        critic_report = self.update_critic(generated_features, lambda_values)

        report = {**model_report, **actor_report, **critic_report}
        return report

    def update_model(
        self,
        batch: Batch,
        key: PRNGKey
    ) -> Tuple[dict, jnp.ndarray]:

        assert self.is_training_instance, "Update function can only be called with training instances."

        def loss(params: hk.Params) -> Tuple[float, dict]:
            _, _, infer, _ = self.model.apply
            outputs_infer = infer(params, key, batch['obs'], batch['action'])
            (prior, posterior), features, decoded, reward, terminal = outputs_infer
            kl = jnp.maximum(tfd.kl_divergence(posterior, prior).mean(), self.config.free_kl)
            obs_f32 = batch['obs'].astype(jnp.float32)
            log_p_obs = decoded.log_prob(obs_f32).mean()
            log_p_rews = reward.log_prob(batch['reward']).mean()
            log_p_terms = terminal.log_prob(batch['terminal']).mean()
            loss_ = self.config.kl_scale * kl - log_p_obs - log_p_rews - log_p_terms

            metrics = {
                'world_model/kl': kl,
                'world_model/post_entropy': posterior.entropy().mean(),
                'world_model/prior_entropy': prior.entropy().mean(),
                'world_model/log_p_obs': -log_p_obs,
                'world_model/log_p_reward': -log_p_rews,
                'world_model/log_p_terminal': -log_p_terms,
                'features': features
            }

            return loss_, metrics

        grads, metrics = jax.grad(loss, has_aux=True)(self.params['model'])
        metrics['world_model/grads'] = optax.global_norm(grads)

        self.params['model'], self.opt_state['model'] = self.grad_step(grads, self.model_optimizer,
                                                                       self.params['model'], self.opt_state['model'])

        return metrics, metrics.pop('features')

    def update_actor(
        self,
        features: jnp.ndarray,
        key: PRNGKey
    ) -> Tuple[dict, Tuple[jnp.ndarray, jnp.ndarray]]:

        assert self.is_training_instance, "Update function can only be called with training instances."

        _, generate_experience, *_ = self.model.apply
        policy = self.actor
        critic = self.critic.apply
        flattened_features = features.reshape((-1, features.shape[-1]))

        def compute_lambda_values(
            next_values: jnp.ndarray,
            rewards: jnp.ndarray,
            terminals: jnp.ndarray,
            discount: float,
            lambda_: float,
        ) -> jnp.ndarray:

            v_lambda = next_values[:, -1] * (1.0 - terminals[:, -1])
            horizon = next_values.shape[1]
            lamda_values = jnp.empty_like(next_values)
            for t in reversed(range(horizon)):
                td = (rewards[:, t] + (1.0 - terminals[:, t]) * (1.0 - lambda_) * discount * next_values[:, t])
                v_lambda = td + v_lambda * lambda_ * discount
                lamda_values = lamda_values.at[:, t].set(v_lambda)
            return lamda_values

        def loss(params: hk.Params):
            generated_features, reward, terminal = generate_experience(self.params['model'], key, flattened_features, policy, params)

            next_values = critic(self.params['critic'], generated_features[:, 1:]).mean()
            lambda_values = compute_lambda_values(next_values,
                                                  reward.mean(),
                                                  terminal.mean(),
                                                  self.config.discount,
                                                  self.config.lambda_)
            discount = discount_(self.config.discount, self.config.imag_horizon - 1)
            loss_ = (-lambda_values * discount).mean()
            return loss_, (generated_features, lambda_values)

        (loss_, aux), grads = jax.value_and_grad(loss, has_aux=True)(self.params['actor'])
        self.params['actor'], self.opt_state['actor'] = self.grad_step(grads, self.actor_optimizer,
                                                                       self.params['actor'], self.opt_state['actor'])

        entropy = policy.apply(self.params['actor'], features[:, 0]).entropy(seed=key).mean()

        metrics = {
            'agent/actor_loss': loss_,
            'agent/actor_grads': optax.global_norm(grads),
            'agent/actor_entropy': entropy
        }

        return metrics, aux

    def update_critic(
        self,
        features: jnp.ndarray,
        lambda_values: jnp.ndarray
    ) -> dict:

        assert self.is_training_instance, "Update function can only be called with training instances."

        def loss(params: hk.Params) -> float:
            values = self.critic.apply(params, features[:, :-1])
            targets = jax.lax.stop_gradient(lambda_values)
            discount = discount_(self.config.discount, self.config.imag_horizon - 1)
            return -(values.log_prob(targets) * discount).mean()

        (loss_, grads) = jax.value_and_grad(loss)(self.params['critic'])
        self.params['critic'], self.opt_state['critic'] = self.grad_step(grads, self.critic_optimizer,
                                                                         self.params['critic'],
                                                                         self.opt_state['critic'])

        metrics = {
            'agent/critic_loss': loss_,
            'agent/critic_grads': optax.global_norm(grads)
        }

        return metrics

    def grad_step(self, grads, optimizer: optax.GradientTransformation, params: hk.Params, opt_state: optax.OptState)\
            -> [hk.Params, optax.OptState]:

        unscaled_grads = self.precision.cast_to_param(grads)
        updates, new_opt_state = optimizer.update(unscaled_grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        grads_finite = jmp.all_finite(unscaled_grads)
        params, opt_state = jmp.select_tree(grads_finite, (new_params, new_opt_state), (params, opt_state))

        return params, opt_state
