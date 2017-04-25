#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:21:32 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np


class SharedGRUCell(tf.contrib.rnn.GRUCell):
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        tf.contrib.rnn.GRUCell.__init__(self, num_units, input_size, activation)
        self.my_scope = None

    def __call__(self, a, b):
        if self.my_scope == None:
            self.my_scope = tf.get_variable_scope()
        else:
            self.my_scope.reuse_variables()
        return tf.nn.rnn_cell.GRUCell.__call__(self, a, b, self.my_scope)

with tf.variable_scope("scope2") as vs:
  cell = SharedGRUCell(10)
  stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
  stacked_cell(tf.Variable(tf.zeros([(20, 10)], name="moo"),
               tf.Variable(tf.zeros([(20, 10)], "bla"))
  # Retrieve just the LSTM variables.
#  vars = [v.name for v in tf.all_variables()
#                    if v.name.startswith(vs.name)]
#  print vars