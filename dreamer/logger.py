'''
from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_evaluation_summary(self, summary, step):
        for k, v in summary.items():
            self.writer.add_scalar(k, float(v), step)
        self.writer.flush()

    def log_metrics(self, summary, step):
        print(f'\n---Training step {step} summary---')
        for k, v in summary.items():
            val = float(v)
            print(f'{k:<40} {val:<.2f}')
            self.writer.add_scalar(k, val, step)
        self.writer.flush()

    # (N, T, C, H, W)
    def log_video(self, images, step=None, name='policy', fps=30):
        self.writer.add_video(name, images, step, fps=fps)
        self.writer.flush()

    def log_images(self, images, step=None, name='policy'):
        self.writer.add_images(name, images, step, dataformats='NHWC')
        self.writer.flush()

    def log_figure(self, figure, step=None, name='policy'):
        self.writer.add_figure(name, figure, step)
        self.writer.flush()
'''

from typing import Dict, Union
from threading import Lock

from tensorboardX import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.lock = Lock()

    def add_scalars(self, scalar_dict: Dict[str, Union[float, int]], step: int):
        with self.lock:
            for k, v in scalar_dict.items():
                self.writer.add_scalar(k, float(v), step)
            self.writer.flush()

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

