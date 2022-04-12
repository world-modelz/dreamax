import json
from typing import Tuple, Dict, Any


class Dataclass:
    def __init__(self, config: Dict[str, Any] = None, load_with_warning: bool = True):
        super().__init__()

        for parent in self.__class__.__mro__[::-1]:
            if hasattr(parent, '__annotations__'):
                self.__dict__.update(
                    {k: parent.__dict__[k] for k in parent.__annotations__.keys()})

        self.update(config, load_with_warning)

    def update(self, config: Dict[str, Any], load_with_warning: bool = True):
        if config is not None:
            for k, v in config.items():
                if isinstance(v, list):
                    v = tuple(v)
                if k in self.__dict__:
                    if isinstance(self.__dict__[k], Dataclass):
                        self.__dict__[k].update(v)
                    else:
                        self.__dict__[k] = v
                else:
                    if load_with_warning:
                        print(
                            f"WARNING: Unknown config parameter {k}={v!r} for section {type(self)}")
                    else:
                        raise ValueError(
                            f"Unknown config parameter {k}={v!r} for section {type(self)}")


class ReplayBufferConfig(Dataclass):
    capacity: int = 200_000
    batch_size: int = 64
    sequence_length: int = 64


class OptimizerConfig(Dataclass):
    lr: float = 6e-4
    eps: float = 1e-7
    clip: float = 100


class RssmConfig(Dataclass):
    hidden: int = 200
    deterministic_size: int = 200
    stochastic_size: int = 30


class EncoderConfig(Dataclass):
    depth: int = 32
    kernels: Tuple[int] = (4, 4, 4, 4)


class DecoderConfig(Dataclass):
    depth: int = 32
    kernels: Tuple[int] = (5, 5, 6, 6)


class OutputHeadConfigBase(Dataclass):
    output_sizes: Tuple[int] = (400, 400)


class ActorConfig(OutputHeadConfigBase):
    min_stddev: float = 1e-4


class DreamerConfiguration(Dataclass):
    log_dir: str = "results"
    seed: int = 0
    task: str = "pendulum.swingup"
    time_limit: int = 1000
    steps: int = 1e6
    prefill: int = 10000
    env_step_per_iter: int = 5
    updates_per_iter: int = 1
    log_every_n_iterations: int = 20
    evaluate_every_n_iterations: int = 100
    episodes_per_evaluate: int = 10
    replay: ReplayBufferConfig = ReplayBufferConfig()
    platform: str = 'gpu'
    jit: bool = True
    render_episodes: int = 1
    evaluate_model: bool = True
    precision: int = 16
    initialization: str = "glorot"
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
        {'output_sizes': (400, 400)})
    terminal: OutputHeadConfigBase = OutputHeadConfigBase(
        {'output_sizes': (400, 400, 400)})
    actor: ActorConfig = ActorConfig({'output_sizes': (400, 400, 400, 400)})
    critic: OutputHeadConfigBase = OutputHeadConfigBase(
        {'output_sizes': (400, 400, 400)})
    actor_opt: OptimizerConfig = OptimizerConfig(
        {'lr': 8e-5, 'eps': 1e-7, 'clip': 100})
    critic_opt: OptimizerConfig = OptimizerConfig(
        {'lr': 8e-5, 'eps': 1e-7, 'clip': 100})


def load_configuration_file(file_path) -> DreamerConfiguration:
    with open(file_path, 'r') as f:
        config = json.load(f)

    return DreamerConfiguration(config)
