
import os
import argparse
import json
import numpy as np
import gym
import functools
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Union

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
from dreamer.utils import Timers, GlobalStepCounter


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


# ToDo: refactor
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

    def __init__(self, config: DreamerConfiguration, env: gym.Env, agent: Dreamer,
                 step_counter: GlobalStepCounter = None, replay_buffer: ReplayBuffer = None,
                 logger: TrainingLogger = None):

        self.config = config
        self.env = env
        self.agent = agent
        self.step_counter = step_counter
        self.replay_buffer = replay_buffer
        self.logger = logger

        self.action_space = self.env.action_space
        self.is_training = self.replay_buffer is not None

        self.reset()

    def reset(self):
        self.steps = 0
        self.obs = self.env.reset()
        self.done = False
        self.sum_reward = 0
        self.episode_steps = 0
        self.state = self.agent.get_inital_state()

    def do_rollout(self, n_steps: int = None, n_episodes: int = None, random: bool = False) -> List[dict]:

        assert n_steps is not None or n_episodes is not None, \
            "Both 'n_steps' and 'n_episodes' are none at least one needs to be set."

        rollout_done = False
        rollout_step_count = 0
        rollout_episode_count = 0

        episodes = []
        episode_summary = defaultdict(list)

        while not rollout_done:
            if random:
                action = self.action_space.sample()
            else:
                action, self.state = self.agent(self.obs, self.state, self.is_training)

            next_obs, reward, self.done, info = self.env.step(action)

            env_transition_dict = dict(obs=self.obs,
                                       next_obs=next_obs,
                                       action=action.astype(np.float32),
                                       reward=np.array(reward, np.float32),
                                       terminal=self.done,
                                       info=info)

            if self.replay_buffer is not None:
                self.replay_buffer.store(env_transition_dict)

            #ToDo: Find a better solution. Don't log it twise.
            episode_summary['obs'].append(self.obs)
            episode_summary['next_obs'].append(next_obs)
            episode_summary['action'].append(action)
            episode_summary['reward'].append(reward)
            episode_summary['terminal'].append(self.done)
            episode_summary['info'].append(info)

            self.obs = next_obs
            self.sum_reward += reward
            rollout_step_count += 1
            self.episode_steps += 1

            if self.is_training and self.step_counter is not None:
                self.step_counter.add_step()

            if self.done:
                rollout_episode_count += 1

                print(f"DONE EPISODE   SUM REWARD: {self.sum_reward}  IN {self.episode_steps} STEPS    RANDOM: {random}")

                if self.is_training and self.logger is not None:
                    metrics = {'env/train/sum_reward': self.sum_reward, 'env/train/episode_len': self.episode_steps}
                    self.logger.add_scalars(metrics, self.step_counter.steps)

                episode_summary['steps'] = [self.episode_steps]
                episodes.append(episode_summary)
                episode_summary = defaultdict(list)

                self.reset()

            # Determine if rollout is done.
            if n_steps is not None:
                if rollout_step_count >= n_steps:
                    rollout_done = True
            if n_episodes is not None:
                if rollout_episode_count >= n_episodes:
                    rollout_done = True

        return episodes


# ToDo: Integrade it better wiht the RolloutWorker
def evaluate(agent, logger, config: DreamerConfiguration, steps, eval_rollout_worker: Rollout_worker):
    evaluation_episodes_summaries = eval_rollout_worker.do_rollout(n_episodes=config.episodes_per_evaluate)

    if config.render_episodes > 0:

        videos = list(map(lambda episode: episode.get('obs'), evaluation_episodes_summaries[:config.render_episodes]))
        videos = np.array(videos, copy=False)
        videos = videos.transpose([0, 1, 4, 2, 3])
        logger.add_video(videos, steps, name='videos/overview')

        more_vidoes = evaluate_model(
            jnp.asarray(evaluation_episodes_summaries[0]['obs']),
            jnp.asarray(evaluation_episodes_summaries[0]['action']),
            next(agent.rng_seq),
            agent.model, agent.model.params,
            get_mixed_precision_policy(config.precision)
        )

        for vid, name in zip(more_vidoes, ('ground_truth', 'reconstructed', 'unroald')):
            logger.add_video(np.array(vid, copy=False).transpose([0, 1, 4, 2, 3]), steps, name='videos/' + name)

    avg_return = np.asarray([sum(episode['reward']) for episode in evaluation_episodes_summaries]).mean()
    avg_len = np.asarray([episode['steps'][0] for episode in evaluation_episodes_summaries]).mean()
    evaluation_episodes_summaries = {'env/eval/sum_reward': avg_return,
                                     'env/eval/episode_len': avg_len}

    logger.add_scalars(evaluation_episodes_summaries, steps)


"""
def train(config: DreamerConfiguration, agent, rollout_worker: Rollout_worker, logger, replay_buffer: ReplayBuffer):
    steps = 0
    iterations = 0
    metrics = defaultdict(float)
    agent_data_path = Path(config.log_dir, 'agent_data')

    if agent_data_path.exists():
        agent.load(agent_data_path)
        steps = agent.training_step
        print(f"Loaded {steps} steps. Continuing training from {config.log_dir}")
    else:
        rollout_worker.do_rollout(n_steps=config.prefill, random=True)

    while steps < config.steps:

        batch_gen = replay_buffer.sample(config.updates_per_iter)

        for _ in range(config.updates_per_iter):
            sample = next(batch_gen)

            agent.learning_states, reports = agent.update(dict(sample), *agent.learning_states, key=next(agent.rng_seq))

            # Average training metrics across update steps.
            for k, v in reports.items():
                metrics[k] += float(v) / (config.updates_per_iter * config.log_every_n_iterations)

        rollout_worker.do_rollout(n_steps=config.env_step_per_iter)

        if iterations != 0 and iterations % config.log_every_n_iterations == 0:
            logger.log_metrics(metrics, steps)
            metrics = defaultdict(float)

        if iterations != 0 and iterations % config.evaluate_every_n_iterations == 0:
            print("Evaluating.")
            evaluation_summaries = evaluate(agent, environment, logger, config, steps)
            training_summary.update(evaluation_summaries)

        steps += config.env_step_per_iter
        iterations += 1

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
"""

def main():
    tf.config.experimental.set_visible_devices([], "GPU")

    args = parse_args()

    config = DreamerConfiguration()
    for config_path in args.configs:
        print(f'Loading configuration: "{config_path}"')
        with open(config_path, 'r') as f:
            json_config = json.load(f)
            config.update(json_config)

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
    environment = create_env(domain, task, config.time_limit, config.seed)
    logger = TrainingLogger(config.log_dir)

    replay_buffer = ReplayBuffer(config=config.replay, obs_space=environment.observation_space,
                                 action_space=environment.action_space,  precision=config.precision, seed=config.seed)

    agent = Dreamer(
        obs_space=environment.observation_space, action_space=environment.action_space,
        model=create_model(config, environment.observation_space),
        actor=create_actor(config, environment.action_space),
        critic=create_critic(config),
        logger=logger, config=config,
        precision=get_mixed_precision_policy(config.precision)
    )

    iterations = 0
    metrics = defaultdict(float)
    agent_data_path = Path(config.log_dir, 'agent_data')
    timers = Timers(['timers/wait_for_data', 'timers/wait_for_rollout', 'timers/wait_for_eval', 'timers/training_time',
                     'timers/iteration_time'])
    step_counter = GlobalStepCounter()

    train_rollout_worker = Rollout_worker(config=config, env=environment, agent=agent, step_counter=step_counter,
                                          replay_buffer=replay_buffer, logger=logger)
    eval_rollout_worker = Rollout_worker(config=config, env=environment, agent=agent)

    if agent_data_path.exists():
        agent.load(agent_data_path)
        steps = agent.training_step
        print(f"Loaded {steps} steps. Continuing training from {config.log_dir}")
    else:
        train_rollout_worker.do_rollout(n_steps=config.prefill, random=True)

    while step_counter.steps < config.steps:
        with timers.timing('timers/iteration_time'):

            with timers.timing('timers/wait_for_data'):
                batch_gen = iter(replay_buffer.sample(config.updates_per_iter))

            with timers.timing('timers/training_time'):
                for _ in range(config.updates_per_iter):
                    sample = next(batch_gen)

                    agent.learning_states, reports = agent.update(dict(sample), *agent.learning_states, key=next(agent.rng_seq))

                    # Average training metrics across update steps.
                    for k, v in reports.items():
                        metrics[k] += float(v) / (config.updates_per_iter * config.log_every_n_iterations)

            with timers.timing('timers/wait_for_rollout'):
                train_rollout_worker.do_rollout(n_steps=config.env_step_per_iter)

            with timers.timing('timers/wait_for_eval'):
                if iterations != 0 and iterations % config.evaluate_every_n_iterations == 0:
                    print("Evaluating.")
                    evaluate(agent, logger, config, step_counter.steps, eval_rollout_worker)

            if iterations != 0 and iterations % config.log_every_n_iterations == 0:
                metrics.update(timers.collect_times())
                logger.add_scalars(metrics, step_counter.steps)
                metrics = defaultdict(float)

            iterations += 1


if __name__ == '__main__':
    main()
