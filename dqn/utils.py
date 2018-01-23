import os

import numpy as np
import tensorflow as tf

def makedir_if_there_is_no(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print("****Directory {} was made".format(path))

def clipped_error(x):
  return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def rgb2gray(img):
  return np.dot(img[...,:3], [0.299, 0.587, 0.114])
