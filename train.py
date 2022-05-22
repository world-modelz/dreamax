import os
import argparse
import json

os.environ['MUJOCO_GL'] = "osmesa"

import numpy as np
import gym
import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk


from dreamer.dreamer import Dreamer, get_mixed_precision_policy
from dreamer.replay_buffer import ReplayBuffer
from dreamer.logger import TrainingLogger
from dreamer.world_model import Actor, DenseDecoder, Decoder, WorldModel
from dreamer.gym_adapter import create_env
from dreamer.configuration import DreamerConfiguration
from train_loop import train


def create_model(config, observation_space):
    def model():
        _model = WorldModel(observation_space, config)

        def filter_state(prev_state, prev_action, observation):
            return _model(prev_state, prev_action, observation)

        def generate_sequence(initial_state, policy, policy_params, actions=None):
            return _model.generate_sequence(initial_state, policy, policy_params, actions)

        def observe_sequence(observations, actions):
            return _model.observe_sequence(observations, actions)

        def decode(feature):
            return _model.decode(feature)

        def init(observations, actions):
            return _model.observe_sequence(observations, actions)

        return init, (filter_state, generate_sequence, observe_sequence, decode)

    return hk.multi_transform(model)


def create_actor(config: DreamerConfiguration, action_space: gym.Space):
    actor = hk.without_apply_rng(hk.transform(
        lambda *obs: Actor(
            config.actor.output_sizes + (2 * np.prod(action_space.shape),),
            config.actor.min_stddev, config.initialization
        )(obs[-1])
    ))
    return actor


def create_critic(config: DreamerConfiguration):
    critic = hk.without_apply_rng(hk.transform(lambda obs: DenseDecoder(
        config.critic.output_sizes + (1,), 'normal', config.initialization)(obs)))
    return critic


def create_agent(config: DreamerConfiguration, environment, logger: TrainingLogger):
    experience = ReplayBuffer(
        config.replay,
        environment.observation_space,
        environment.action_space,
        config.precision,
        config.seed)
    agent = Dreamer(
        environment.observation_space,
        environment.action_space,
        create_model(config, environment.observation_space),
        create_actor(config, environment.action_space),
        create_critic(config),
        experience,
        logger,
        config,
        get_mixed_precision_policy(config.precision)
    )
    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+')
    args = parser.parse_args()
    return args


def main():
    tf.config.experimental.set_visible_devices([], "GPU")

    args = parse_args()

    config = DreamerConfiguration()
    if args.configs:
        for config_path in args.configs:
            print(f'Loading configuration: "{config_path}"')
            with open(config_path, 'r') as f:
                json_config = json.load(f)
                config.update(json_config)

    np.random.seed(config.seed)

    if config.platform:
        jax.config.update('jax_platform_name', config.platform)

    print('Available devices:')
    for device in jax.devices():
        print(device)

    if not config.jit:
        jax.config.update('jax_disable_jit', True)

    if config.precision == 16:
        policy = get_mixed_precision_policy(16)
        hk.mixed_precision.set_policy(WorldModel, policy)
        hk.mixed_precision.set_policy(Actor, policy)
        hk.mixed_precision.set_policy(DenseDecoder, policy)
        hk.mixed_precision.set_policy(Decoder, policy.with_output_dtype(jnp.float32))

    domain, task = config.task.split('.')
    environment = create_env(domain, task, config.time_limit, config. action_repeat, config.seed)
    logger = TrainingLogger(config.log_dir)
    agent = create_agent(config, environment, logger)
    train(config, agent, environment, logger)


if __name__ == '__main__':
    main()
