from typing import Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import optax

from dreamer.configuration import OptimizerConfig

PRNGKey = jnp.ndarray
LearningState = Tuple[hk.Params, optax.OptState]


class Learner:
    def __init__(
        self,
        model: Union[hk.Transformed, hk.MultiTransformed],
        seed: PRNGKey,
        optimizer_config: OptimizerConfig,
        precision: jmp.Policy,
        *input_example: Tuple
    ):
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(optimizer_config.clip),
            optax.scale_by_adam(eps=optimizer_config.eps),
            optax.scale(-optimizer_config.lr),
        )
        self.model = model
        self.params = self.model.init(seed, *input_example)
        self.opt_state = self.optimizer.init(self.params)
        self.precision = precision

    @property
    def apply(self) -> Union[Callable, Tuple[Callable]]:
        return self.model.apply

    @property
    def learning_state(self):
        return self.params, self.opt_state

    @learning_state.setter
    def learning_state(self, state):
        self.params = state[0]
        self.opt_state = state[1]

    def grad_step(self, grads, state: LearningState):
        params, opt_state = state
        unscaled_grads = self.precision.cast_to_param(grads)
        updates, new_opt_state = self.optimizer.update(
            unscaled_grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        grads_finite = jmp.all_finite(unscaled_grads)
        new_params, new_opt_state = jmp.select_tree(
            grads_finite, (new_params, new_opt_state), (params, opt_state)
        )
        return new_params, new_opt_state
