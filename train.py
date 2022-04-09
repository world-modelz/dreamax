
import os
import argparse
import json
import numpy as np
import gym
import functools
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

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


def create_model(config, obs_space):
    def model():
        _model = WorldModel(obs_space, config)

        def filter_state(prev_state, prev_action, obs):
            return _model(prev_state, prev_action, obs)

        def generate_sequence(initial_state, policy, policy_params, actions=None):
            return _model.generate_sequence(initial_state, policy, policy_params, actions)

        def observe_sequence(obss, actions):
            return _model.observe_sequence(obss, actions)

        def decode(feature):
            return _model.decode(feature)

        def init(obss, actions):
            return _model.observe_sequence(obss, actions)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=['configs/dreamer_v2.json'])
    args = parser.parse_args()
    return args


@functools.partial(jax.jit, static_argnums=(3, 5))
def evaluate_model(obss, actions, key, model, model_params, precision):
    length = min(len(obss) + 1, 50)
    obss, actions = jax.tree_map(lambda x: x.astype(precision.compute_dtype), (obss, actions))
    _, generate_sequence, infer, decode = model.apply
    key, subkey = jax.random.split(key)
    _, features, infered_decoded, *_ = infer(model_params, subkey, obss[None, :length], actions[None, :length])
    conditioning_length = length // 5
    key, subkey = jax.random.split(key)
    generated, *_ = generate_sequence(
        model_params, subkey,
        features[:, conditioning_length], None, None,
        actions=actions[None, conditioning_length:])
    key, subkey = jax.random.split(key)
    generated_decoded = decode(model_params, subkey, generated)
    out = (obss[None, conditioning_length:length],
           infered_decoded.mean()[:, conditioning_length:length],
           generated_decoded.mean())
    out = jax.tree_map(lambda x: ((x + 0.5) * 255).astype(jnp.uint8), out)
    return out

'''
def do_episode(agent, training: bool, environment, config: DreamerConfiguration, pbar, render: bool):
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    obs = environment.reset()
    while not done:
        action = agent(obs, training)
        next_obs, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated', False)

        if training:
            agent.observe(dict(obs=obs,
                               next_obs=next_obs,
                               action=action.astype(np.float32),
                               reward=np.array(reward, np.float32),
                               terminal=np.array(terminal, np.float32),
                               info=info))

        episode_summary['obs'].append(obs)
        episode_summary['next_obs'].append(next_obs)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        obs = next_obs

        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))

        pbar.update(config.action_repeat)
        steps += config.action_repeat

    episode_summary['steps'] = [steps]
    return steps, episode_summary
'''

'''
def interact(agent, environment, steps, config: DreamerConfiguration, training=True, on_episode_end=None):
    pbar = tqdm(total=steps)
    steps_count = 0
    episodes = []
    while steps_count < steps:
        render = len(episodes) < config.render_episodes and not training
        episode_steps, episode_summary = do_episode(agent, training, environment, config, pbar, render)
        steps_count += episode_steps
        episodes.append(episode_summary)
        if on_episode_end is not None:
            on_episode_end(episode_summary, steps_count)
    pbar.close()
    return steps, episodes
'''


class Rollout_worker():

    def __init__(self, config: DreamerConfiguration, env: gym.Env, agent_forward_fn: callable, replay_buffer: ReplayBuffer = None):
        self.config = config
        self.env = env
        self.agent_forward_fn = agent_forward_fn
        self.replay_buffer = replay_buffer

        self.action_space = self.env.action_space
        self.is_training = self.replay_buffer is not None

        self.reset()

    def reset(self):
        self.steps = 0
        self.obs = self.env.reset()
        self.done = False
        self.sum_reward = 0
        self.episode_steps = 0

    def do_rollout(self, n_steps: int = None, n_episodes: int = None, random: bool = False):

        assert n_steps is not None or n_episodes is not None, "Both 'n_steps' and 'n_episodes' are none at least one needs to be set."

        rollout_done = False
        rollout_step_count = 0
        rollout_episode_count = 0

        while not rollout_done:
            if random:
                action = self.action_space.sample()
            else:
                action = self.agent_forward_fn(self.obs, self.is_training)

            next_obs, reward, self.done, info = self.env.step(action)

            env_transition_dict = dict(obs=self.obs,
                                       next_obs=next_obs,
                                       action=action.astype(np.float32),
                                       reward=np.array(reward, np.float32),
                                       terminal=self.done,
                                       info=info)

            if self.replay_buffer is not None:
                self.replay_buffer.store(env_transition_dict)

            self.obs = next_obs
            self.sum_reward += reward
            rollout_step_count += 1
            self.episode_steps += 1

            if self.done:
                rollout_episode_count += 1

                print(f"DONE EPISODE   SUM REWARD: {self.sum_reward}  IN {self.episode_steps} STEPS    RANDOM: {random}")

                # todo: LOG END OG EPISODE

                self.reset()

            # Determine if rollout is done.
            if n_steps is not None:
                if rollout_step_count >= n_steps:
                    rollout_done = True
            if n_episodes is not None:
                if rollout_episode_count >= n_episodes:
                    rollout_done = True


def make_summary(summaries, prefix):
    avg_return = np.asarray([sum(episode['reward']) for episode in summaries]).mean()
    avg_len = np.asarray([episode['steps'][0] for episode in summaries]).mean()
    epoch_summary = {prefix + '/average_return': avg_return,
                     prefix + '/average_episode_length': avg_len}
    return epoch_summary


def evaluate(agent, train_env, logger, config: DreamerConfiguration, steps):
    evaluation_steps, evaluation_episodes_summaries = interact(
        agent, train_env, config.evaluation_steps_per_epoch, config, training=False)

    if config.render_episodes:
        videos = list(map(lambda episode: episode.get('image'), evaluation_episodes_summaries[:config.render_episodes]))
        logger.log_video(np.array(videos, copy=False).transpose([0, 1, 4, 2, 3]), steps, name='videos/overview')

    if config.evaluate_model:
        more_vidoes = evaluate_model(
            jnp.asarray(evaluation_episodes_summaries[0]['obs']),
            jnp.asarray(evaluation_episodes_summaries[0]['action']),
            next(agent.rng_seq),
            agent.model, agent.model.params,
            get_mixed_precision_policy(config.precision)
        )

        for vid, name in zip(more_vidoes, ('gt', 'infered', 'generated')):
            logger.log_video(np.array(vid, copy=False).transpose([0, 1, 4, 2, 3]), steps, name='videos/' + name)

    return make_summary(evaluation_episodes_summaries, 'evaluation')


def on_episode_end(episode_summary, logger, global_step, steps_count):
    episode_return = sum(episode_summary['reward'])
    steps = global_step + steps_count
    print(f"\nFinished episode with return: {episode_return:.2f}")
    summary = {'training/episode_return': episode_return}
    logger.log_evaluation_summary(summary, steps)


def train(config: DreamerConfiguration, agent, rollout_worker: Rollout_worker, logger):
    steps = 0
    agent_data_path = Path(config.log_dir, 'agent_data')

    if agent_data_path.exists():
        agent.load(agent_data_path)
        steps = agent.training_step
        print(f"Loaded {steps} steps. Continuing training from {config.log_dir}")
    else:
        rollout_worker.do_rollout(n_steps=config.prefill, random=True)

    while steps < config.steps:
        agent.update()

        rollout_worker.do_rollout(n_steps=config.train_every)
        agent.update()

        '''
        print("Performing a training epoch.")

        _on_episode_end = lambda episode_summary, steps_count: on_episode_end(episode_summary, logger=logger, global_step=steps,
                                                                              steps_count=steps_count)

        training_steps, training_episodes_summaries = interact(
            agent, environment, config.training_steps_per_epoch, config, training=True, on_episode_end=_on_episode_end)

        steps += training_steps
        training_summary = make_summary(training_episodes_summaries, 'training')

        if config.evaluation_steps_per_epoch:
            print("Evaluating.")
            evaluation_summaries = evaluate(agent, environment, logger, config, steps)
            training_summary.update(evaluation_summaries)

        logger.log_evaluation_summary(training_summary, steps)
        # agent.save(agent_data_path)
        '''


def main():
    tf.config.experimental.set_visible_devices([], "GPU")

    args = parse_args()

    config = DreamerConfiguration()
    for config_path in args.configs:
        print(f'Loading configuration: "{config_path}"')
        with open(config_path, 'r') as f:
            json_config = json.load(f)
            config.update(json_config, load_with_warning=False)

    np.random.seed(config.seed)

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

    replay_buffer = ReplayBuffer(config=config.replay, obs_space=environment.observation_space,
                                 action_space=environment.action_space,  precision=config.precision, seed=config.seed)

    agent = Dreamer(
        obs_space=environment.observation_space, action_space=environment.action_space,
        model=create_model(config, environment.observation_space),
        actor=create_actor(config, environment.action_space),
        critic=create_critic(config),
        replay_buffer=replay_buffer,
        logger=logger, config=config,
        precision=get_mixed_precision_policy(config.precision)
    )

    rollout_worker = Rollout_worker(config=config, env=environment, agent_forward_fn=agent.__call__,
                                    replay_buffer=replay_buffer)

    train(config=config, agent=agent, rollout_worker=rollout_worker, logger=logger)


if __name__ == '__main__':
    main()
