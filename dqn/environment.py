import gym

import cv2

from .utils import rgb2gray

import random

class Environment:
  def __init__(self, config):
    self.env_name = config.env_name
    self.make()

    self.action_repeat= config.action_repeat
    self.no_op_max = config.no_op_max
    self.rendering = config.render
    self.in_shape = (config.in_height, config.in_width)

  def initialize_game(self):
    if self.lives == 0:
      self._screen = self.env.reset()
    self._step(0)
    self.render()
    return self.screen, 0, 0, self.done

  def start_randomly(self):
    self.initialize_game()
    for _ in range(random.randint(0, self.no_op_max)):
      self._step(0)
    self.render()
    return self.screen, 0, 0, self.done

  def _step(self, action):
    self._screen, self.reward, self.done, self.info = self.env.step(action)

  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @property
  def screen(self):
    return cv2.resize(rgb2gray(self._screen)/255., self.in_shape)

  @property
  def action_space_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    if self.info is None:
      return 0
    else:
      return self.info["ale.lives"]

  def render(self):
    if self.rendering:
      self.env.render()

  def after_act(self):
    self.render()

  def act(self, action):
    action_reward = 0
    before_lives = self.lives

    for _ in range(self.action_repeat):
      self._step(action)
      action_reward += self.reward

      if before_lives > self.lives:
        action_reward = -1
        self.done = True

      if self.done:
        break

    self.reward = action_reward

    self.after_act(action)
    return self.screen, self.reward, self.done

  def make(self):
    self.env = gym.make(self.env_name)
    self._screen = None
    self.reward = 0
    self.done = True
    self.info = None
