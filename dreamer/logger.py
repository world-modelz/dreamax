from typing import Dict, Union
from threading import Lock

from tensorboardX import SummaryWriter

#SOME HOW IT NEVER INSTALES FOR ME IN DOCKER.
#ToDo: Fix it not being in docker despite it being in the requirements.txt
import pip
pip.main(['install', 'moviepy'])


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
