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
    if config.test:
        agent.play(test=True)
    elif config.train:
        agent.train()
    else:
        agent.play()


if __name__ == "__main__":
    main()
