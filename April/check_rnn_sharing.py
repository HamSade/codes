#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 14:56:25 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np

class SharedGRUCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        tf.contrib.rnn.BasicLSTMCell.__init__(self, num_units, input_size, activation)
        self.my_scope = None

    def __call__(self, a, b):
        if self.my_scope == None:
            self.my_scope = tf.get_variable_scope()
        else:
            self.my_scope.reuse_variables()
        return tf.contrib.rnn.BasicLSTMCell.__call__(self, a, b, self.my_scope)

with tf.variable_scope("scope2") as vs:
  cell = SharedGRUCell(10)
  stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
  stacked_cell(tf.Variable(np.zeros((20, 10), dtype=np.float32), name="moo"),
               tf.Variable(np.zeros((20, 10), dtype=np.float32), "bla"))
  # Retrieve just the LSTM variables.
  vars = [v.name for v in tf.all_variables()
                    if v.name.startswith(vs.name)]
  print(vars)