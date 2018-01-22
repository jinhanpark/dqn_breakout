import tensorflow as tf

from .model import DQN
from .replay_memory import ReplayMemory

from .utils import makedir_if_there_is_no

class Agent(DQN):
  def __init__(self, sess, config, env):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.env = env
    self.memory = ReplayMemory(self.config)

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.get_summary_ops()
    self._build_model()
    self.make_directories()

  def make_directories(self):
    dirs = [self.config.mem_dir, self.config.log_dir, self.config.ckpt_dir]
    for d in dirs:
      makedir_if_there_is_no(d)

  def get_summary_ops(self):
    with tf.variable_scope("summary"):
      summary_tags = ["average.reward", "average.loss", "average.Q", \
                      "episode.avg_reward", "episode.max_reward", \
                      "training.learning_rate"]
      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in summary_tags:
        self.summary_placeholders[tag] = tf.placeholder(tf.float32, None, name=tag)
        self.summary_ops[tag] = tf.summary.scalar("{}/{}".format(self.config.env_name, tag),
                                                  self.summary_placeholders[tag])

  def summarize(self, tag_dict, step=None):
    summary_lst = self.sess.run(
      [self.summary_ops[tag] for tag in tag_dict.keys()],
      {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
    for elt in summary_lst:
      self.writer.add_summary(summary_str, step)
    print("****summarized")

  def train(self):
    self.initialize_variables_and_copy_network()
    
    self.make_directories()
    if self.config.load_ckpt:
      if self.load():
        try:
          self.memory.load()
        except:
          print("****FAILED to load memory")
      else:
        print("****FAILED to load checkpoint and memory")

    start_step = self.sess.run(self.global_step)

    # for step in tqdm(range(start_step, self.config.max_step), ncols=70):
    #   if self.step == self.learn_start:
    #     pass
