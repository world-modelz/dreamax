from typing import Dict, Union
from threading import Lock

import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.lock = Lock()

        self.scalars_buffer = defaultdict(list)

    def flush_scalars(self, step: int, do_print: bool = False):
        with self.lock:
            self.scalars_buffer = {k: float(np.mean(v)) for k, v in self.scalars_buffer.items()}

            if do_print:

                keys = list(self.scalars_buffer.keys())
                keys.sort()
                last_first_key = keys[0].split('/')[0]

                print('=' * 60)
                print(f"|                 env step: {step}")
                print('=' * 60)
                for k in keys:

                    if k.split('/')[0] != last_first_key:
                        last_first_key = k.split('/')[0]
                        print('')

                    print(f"{(k + ':'): <32} {self.scalars_buffer[k]}")

                print('=' * 60)
                print('')

            for k, v in self.scalars_buffer.items():
                self.writer.add_scalar(k, float(v), step)

            self.writer.flush()
            self.scalars_buffer = defaultdict(list)



    def add_scalars(self, scalar_dict: Dict[str, Union[float, int]]):
        with self.lock:
            for k, v in scalar_dict.items():
                self.scalars_buffer[k].append(float(v))

    # (N, T, C, H, W)
    def add_video(self, images, step=None, name='policy', fps=30):
        with self.lock:
            self.writer.add_video(name, images, step, fps=fps)
            self.writer.flush()

    def add_images(self, images, step=None, name='policy'):
        with self.lock:
            self.writer.add_images(name, images, step, dataformats='NHWC')
            self.writer.flush()

    def add_figure(self, figure, step=None, name='policy'):
        with self.lock:
            self.writer.add_figure(name, figure, step)
            self.writer.flush()

