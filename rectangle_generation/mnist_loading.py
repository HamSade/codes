#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:20:06 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
from time import time
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


#%%

#sess = tf.InteractiveSession()

#x = tf.placeholder(tf.float32, shape=[None, 784])
#y_ = tf.placeholder(tf.float32, shape=[None, 10])

x = mnist.train.images[:1,:]
#batch = mnist.train.next_batch(1)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

sample = sess.run(x)

print(sample.shape)
plt.pcolormesh(sample)

#train_step.run(feed_dict={x: batch[0], y_: batch[1]})