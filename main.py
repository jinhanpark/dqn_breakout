import tensorflow as tf
import numpy as np

from config import Config
from dqn.agent import Agent
from dqn.environment import Environment

def main():
  sess = tf.Session()
  config = Config()
  env = Environment(config)
  agent = Agent(sess, config, env)
  agent.train()
  agent.save()
  agent.summarize({})


if __name__=="__main__":
  main()
