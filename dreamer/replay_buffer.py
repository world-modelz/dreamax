from typing import Dict, Union, Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from gym.spaces import Space
from tf_agents.replay_buffers import episodic_replay_buffer

from dreamer.configuration import ReplayBufferConfig

Transition = Dict[str, Union[np.ndarray, dict]]
Batch = Dict[str, np.ndarray]


def preprocess(image):
    return image / 255.0 - 0.5


def quantize(image):
    return ((image + 0.5) * 255).astype(np.uint8)


class ReplayBuffer:
    def __init__(
            self,
            config: ReplayBufferConfig,
            obs_space: Space,
            action_space: Space,
            precision: int,
            seed: int):

        self.config = config

        dtype = {16: tf.float16, 32: tf.float32}[precision]

        data_spec = {
            'obs': tf.TensorSpec(obs_space.shape, tf.uint8),
            'action': tf.TensorSpec(action_space.shape, dtype),
            'reward': tf.TensorSpec((), dtype),
            'terminal': tf.TensorSpec((), dtype)
        }

        self.buffer = episodic_replay_buffer.EpisodicReplayBuffer(data_spec, seed=seed, capacity=config.capacity,
                                                                  buffer_size=1, dataset_drop_remainder=True,
                                                                  completed_only=False, begin_episode_fn=lambda _: True,
                                                                  end_episode_fn=lambda _: True)

        self.current_episode = {'obs': [], 'action': [], 'reward': [], 'terminal': []}
        self.idx = 0
        self.dtype = dtype
        ds = self.buffer.as_dataset(self.config.batch_size, self.config.sequence_length + 1)
        ds = ds.map(self._preprocess, tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(10)
        self.dataset = ds

    def _preprocess(self, episode, _):
        episode['obs'] = preprocess(tf.cast(episode['obs'], self.dtype))
        # shift obs, terminals and reward by one timestep, since
        # RSSM uses the *previous* action and state together with the
        # current obs to infer the *current* state
        for k in ['obs', 'terminal', 'reward']:
            episode[k] = episode[k][:, 1:]

        episode['action'] = episode['action'][:, :-1]

        return episode

    def store(self, transition: Transition):
        episode_end = (transition['terminal'] or transition['info'].get('TimeLimit.truncated', False))
        for k, v in self.current_episode.items():
            v.append(transition[k])

        if episode_end:
            self.current_episode['obs'].append(transition['next_obs'])
            episode = {k: np.asarray(v) for k, v in self.current_episode.items()}
            episode['obs'] = quantize(episode['obs'])
            new_idx = self.buffer.add_sequence(episode, tf.constant(self.idx, tf.int64))
            self.idx = int(new_idx)
            self.current_episode = {k: [] for k in self.current_episode.keys()}

    def sample(self, n_batches: int) -> Iterator[Batch]:
        return tfds.as_numpy(self.dataset.take(n_batches))
