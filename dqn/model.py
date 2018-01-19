import os

import tensorflow as tf

class BaseModel:
  def __init__(self, config):
    self.config = config
    self.saver = tf.train.Saver()

  def save(self):
    model_name = self.config.model_name
    ckpt_dir = os.path.join(self.config.ckpt_dir, model_name)
    self.saver.save(self.sess, ckpt_dir, global_step=self.sess.run(self.global_step))
    print("****weights saved")

  def _load(self):
    print("****Trying to load checkpoint")
    ckpt = tf.train.get_checkpoint_state(self.config.ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_path = ckpt.model_checkpoint_path
      self.saver.restore(self.sess, ckpt_path)
      print("****Weights are restored, fixed net must be updated")
      return True
    else:
      return False

  def load(self):
    return self._load()

class DQN(BaseModel):
  def __init__(self, config):
    super(DQN, self).__init__(config)

  def _build_model(self):
    self.S_in = tf.placeholder(tf.float32, (None, self.config.in_height, self.config.input_width, self.config.history_length), name="state")
    self.Q = self._q_net_cnn("train")
    self.fixed_Q = self._q_net_cnn("fixed")
    self.train_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="train")
    self.fixed_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope="fixed")

    self.loss, self.train_op = self._loss_and_train_op()
    self.copy_ops = self._var_copy_ops()

    self.sess.run(tf.global_variables_initializer())

  def initialize_variables_and_copy_network(self):
    self.sess.run(tf.global_variables_initializer())
    self.update_fixed_target()

  def _q_net_cnn(self, scope):
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    h = self.S_in
    with tf.variable_scope(scope):
      for i, (num_filters, kernel_size, stride) in enumerate(self.config.cnn_archi):
        h = tf.layers.conv2d(
          inputs=mid,
          filters=num_filters,
          kernel_size=kernel_size,
          strides=stride,
          activation=activation_fn,
          kernel_initializer=initializer,
          name="conv{}".format(i + 1))
      h = tf.layers.flatten(h)
      for i, dim in enumerate(self.config.fc_archi):
        h = tf.layers.dense(
          inputs=mid,
          units=dim,
          activation=activation_fn,
          kernel_initializer=initializer
          name="fc{}".format(i + 1))
      Q = tf.layers.dense(
        inputs=h,
        units=self.config.action_space_size,
        kernel_initializer=initializer
        name="Q")
    return Q

  def _loss_and_train_op(self):
    with tf.variable_scope("optimizer"):
      self.A_in = tf.placeholder(tf.int32, (None,), name="action")
      self.target = tf.placeholder(tf.float32, (None,), name="target_q")
      a_one_hot = tf.one_hot(self.A_in, self.config.action_space_size, 1., 0., name="one_hot")
      Q_batch = tf.reduce_sum(tf.multiply(a_one_hot, self.Q), axis=1, name="q_batch")
      error = self.target - Q_batch
      loss = tf.reduce_mean(clipped_error(error), name="loss")
      slef.lr_step = tf.placeholder(tf.int64, None, name="lr_step")
      self.decayed_lr = tf.maximum(
        self.config.lr_min,
        tf.train.exponential_decay(
          self.config.lr,
          self.lr_step,
          self.config.lr_decay_step,
          self.config.lr_decay,
          staircase=True))
      train_op = tf.train.RMSPropOptimizer(
        self.decayed_lr, momentum=0.95, epsilon=0.01).minimize(loss)
    return loss, train_op

  def _var_copy_ops(self):
    with tf.variable_scope("copy_ops"):
      ops = []
      for t_var, f_var in zip(self.train_vars, self.fixed_vars):
        print("variable name matching for debug : ", t_var, ", ", f_var)
        ops.append(f_var.assign(t_var.value()))
    return ops

  def update_fixed_target(self):
    self.sess.run(self.copy_ops)
    print("fixed target updated")

  def load(self):
    rt = self._load()
    self.update_fixed_target()
    return rt
