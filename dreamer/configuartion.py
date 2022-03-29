import json
import dataclasses
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ReplayBufferConfig:
    capacity: int = 1000,
    batch: int = 32
    sequence_length: int = 50


@dataclass
class OptimizerConfig:
    lr: float = 6e-4,
    eps: float = 1e-7,
    clip: float = 100


@dataclass
class RssmConfig:
    hidden: int = 200,
    deterministic_size: int = 200
    stochastic_size: int = 30


@dataclass
class EncoderConfig:
    depth: int = 32
    kernels: Tuple[int] = (4, 4, 4, 4)


@dataclass
class DecoderConfig:
    depth: int = 32
    kernels: Tuple[int] = (5, 5, 6, 6)


@dataclass
class OutputHeadConfigBase:
    output_sizes: Tuple[int] = (400, 400)


@dataclass
class ActorConfig(OutputHeadConfigBase):
    min_stddev: float = 1e-4


@dataclass
class DreamerConfiguration:
    log_dir: str = "results"
    seed: int = 0
    task: str = "pendulum.swingup"
    time_limit: int = 1000
    action_repeat: int = 2
    steps: int = 1e6
    training_steps_per_epoch: int = 2.5e4
    evaluation_steps_per_epoch: int = 1e4
    prefill: int = 5000
    train_every: int = 1000
    update_steps: int = 100
    replay: ReplayBufferConfig = ReplayBufferConfig()
    jit: bool = True
    render_episodes: int = 0
    evaluate_model: bool = True,
    precision: int = 16,
    initialization: str = "glorot",
    rssm: RssmConfig = RssmConfig()
    model_opt: OptimizerConfig = OptimizerConfig()
    discount: float = 0.99
    lambda_: float = 0.95
    imag_horizon: int = 15
    free_kl: float = 3.0
    kl_scale: float = 1.0

    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()

    reward: OutputHeadConfigBase = OutputHeadConfigBase(
        output_sizes=(400, 400))
    terminal: OutputHeadConfigBase = OutputHeadConfigBase(
        output_sizes=(400, 400, 400))

    actor: ActorConfig = ActorConfig(output_sizes=(400, 400, 400, 400))

    critic: OutputHeadConfigBase = OutputHeadConfigBase(
        output_sizes=(400, 400, 400))

    actor_opt: OptimizerConfig = OptimizerConfig(lr=8e-5, eps=1e-7, clip=100)
    critic_opt: OptimizerConfig = OptimizerConfig(lr=8e-5, eps=1e-7, clip=100)


def read_section(data: dict, config_type: type = DreamerConfiguration):
    fields = {f.name: f for f in dataclasses.fields(config_type)}
    d = config_type()
    for k, v in data.items():
        f = fields.get(k)
        if f is None:
            raise RuntimeError(f'Configuration value "{k}" not supported.')
        if dataclasses.is_dataclass(f.type):
            v = read_section(v, f.type)
        d.__setattr__(k, v)
    return d


def load_configuration_file(file_path) -> dict[str, DreamerConfiguration]:
    with open(file_path, 'r') as f:
        data = json.load(f)
        return {config_name: read_section(data[config_name]) for config_name in data}
