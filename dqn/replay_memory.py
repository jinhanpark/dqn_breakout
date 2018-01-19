import random
import numpy as np

class ReplayMemory:
  def __init__(self, config):
    self.save_dir = config.mem_dir

    self.memory_size = config.memory_size

    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.screens = np.empty((self.memory_size, config.in_height, config.in_width), dtype = np.float16)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.dones = np.empty(self.memory_size, dtype = np.bool)
    self.counts = np.empty(2, dtype = int)

    self.history_length = config.history_length
    self.shape = (config.in_height, config.in_width)
    self.batch_size = config.batch_size

    self.count = 0
    self.current = 0
    self.states = np.empty((self.batch_size, self.history_length) + self.shape, dtype = np.float16)
    self.next_states = np.empty((self.batch_size, self.history_length) + self.shape, dtype = np.float16)

  def memorize(self, action, screen, reward, done):
    assert screen.shape == self.shape
    self.actions[self.current] = action
    self.screens[self.current] = screen
    self.rewards[self.current] = reward
    self.dones[self.current] = done
    self.count = min(self.memory_size, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def _state(self, index):
    assert self.count > self.history_length
    index = index % self.count
    if index > self.history_length - 1:
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def get_batch(self):
    assert self.count > self.history_length + 1
    indexes = []
    while len(indexes) < self.batch_size:
      while True:
        index = random.randint(self.history_length, self.count - 1)
        if self.current <= index or index - self.history_length + 1 <= self.current:
          continue
        if self.dones[(index - self.history_length):index].any():
          continue
        break
      self.states[len(indexes), ...] = self._state(index - 1)
      self.next_states[len(indexes), ...] = self._state(index)
      indexes.append(index)
    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    dones = self.dones[indexes]
    return self.states, actions, rewards, self.next_states, dones

  def _store_counts(self):
    self.counts[0] = self.count
    self.counts[1] = self.current

  def _restore_counts(self):
    self.count = self.counts[0]
    self.current = self.counts[1]

  def save(self):
    for name, array in\
        zip(["actions", "rewards", "screens", "terminals", "counts"],
            [self.actions, self.rewards, self.screens, self.terminals, self.counts]):
      save_npy(array, os.path.join(self.save_dir, name))

  def load(self):
    for name, array in\
        zip(["actions", "rewards", "screens", "terminals", "counts"],
            [self.actions, self.rewards, self.screens, self.terminals, self.counts]):
      array = load_npy(os.path.join(self.save_dir, name))

def save_npy(obj, path):
  np.save(path, obj)
  print("****memory saved: {}".format(path))

def load_npy(path):
  obj = np.load(path)
  print("****memory loaded: {}".format(path))
