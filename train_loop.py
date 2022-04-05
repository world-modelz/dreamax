from collections import defaultdict
import functools
from pathlib import Path
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np

from dreamer.configuration import DreamerConfiguration
from dreamer.dreamer import get_mixed_precision_policy


@functools.partial(jax.jit, static_argnums=(3, 5))
def evaluate_model(observations, actions, key, model, model_params, precision):
    length = min(len(observations) + 1, 50)
    observations, actions = jax.tree_map(lambda x: x.astype(precision.compute_dtype), (observations, actions))
    _, generate_sequence, infer, decode = model.apply
    key, subkey = jax.random.split(key)
    _, features, infered_decoded, *_ = infer(model_params, subkey, observations[None, :length], actions[None, :length])
    conditioning_length = length // 5
    key, subkey = jax.random.split(key)
    generated, *_ = generate_sequence(
        model_params, subkey,
        features[:, conditioning_length], None, None,
        actions=actions[None, conditioning_length:])
    key, subkey = jax.random.split(key)
    generated_decoded = decode(model_params, subkey, generated)
    out = (observations[None, conditioning_length:length],
           infered_decoded.mean()[:, conditioning_length:length],
           generated_decoded.mean())
    out = jax.tree_map(lambda x: ((x + 0.5) * 255).astype(jnp.uint8), out)
    return out


def do_episode(agent, training: bool, environment, config: DreamerConfiguration, pbar, render: bool):
    episode_summary = defaultdict(list)
    steps = 0
    done = False
    observation = environment.reset()
    while not done:
        action = agent(observation, training)
        next_observation, reward, done, info = environment.step(action)
        terminal = done and not info.get('TimeLimit.truncated', False)
        if training:
            agent.observe(dict(observation=observation,
                               next_observation=next_observation,
                               action=action.astype(np.float32),
                               reward=np.array(reward, np.float32),
                               terminal=np.array(terminal, np.float32),
                               info=info))
        episode_summary['observation'].append(observation)
        episode_summary['next_observation'].append(next_observation)
        episode_summary['action'].append(action)
        episode_summary['reward'].append(reward)
        episode_summary['terminal'].append(terminal)
        episode_summary['info'].append(info)
        observation = next_observation
        if render:
            episode_summary['image'].append(environment.render(mode='rgb_array'))
        pbar.update(config.action_repeat)
        steps += config.action_repeat
    episode_summary['steps'] = [steps]
    return steps, episode_summary


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
            jnp.asarray(evaluation_episodes_summaries[0]['observation']),
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
    print("\nFinished episode with return: {}".format(episode_return))
    summary = {'training/episode_return': episode_return}
    logger.log_evaluation_summary(summary, steps)


def train(config: DreamerConfiguration, agent, environment, logger):
    steps = 0
    agent_data_path = Path(config.log_dir, 'agent_data')
    if agent_data_path.exists():
        agent.load(agent_data_path)
        steps = agent.training_step
        print(f"Loaded {steps} steps. Continuing training from {config.log_dir}")
    while steps < config.steps:
        print("Performing a training epoch.")
        training_steps, training_episodes_summaries = interact(
            agent, environment, config.training_steps_per_epoch, config,
            training=True,
            on_episode_end=lambda episode_summary, steps_count: on_episode_end(
                episode_summary, logger=logger, global_step=steps,
                steps_count=steps_count))
        steps += training_steps
        training_summary = make_summary(training_episodes_summaries, 'training')
        if config.evaluation_steps_per_epoch:
            print("Evaluating.")
            evaluation_summaries = evaluate(agent, environment, logger, config,
                                            steps)
            training_summary.update(evaluation_summaries)
        logger.log_evaluation_summary(training_summary, steps)
        # agent.save(agent_data_path)
    environment.close()
    return agent
