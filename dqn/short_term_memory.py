import numpy as np

class ShortTermMemory:
  def __init__(self, config):
    self.frames = np.zeros([config.in_height, config.in_width, config.history_length],
                           dtype=np.float32)

  def add(self, frame):
    self.frames[:, :, :-1] = self.frames[:, :, 1:]
    self.frames[:, :, -1] = frame

  def reset(self):
    self.history *= 0
