#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:15:12 2017

@author: hsadeghi
"""
import tflearn 
import tensorflow as tf



input_data = tflearn.input_data(shape=[None, 1,784])


rnn1 = tflearn.layers.recurrent.lstm (input_data, 512, name='rnn1')

fc1_weights_var = rnn1.W
fc1_biases_var = rnn1.b

init = tf.global_variables_initializer()

sess=tf.Session()

sess.run(init)

sess.run(fc1_weights_var)
print(fc1_weights_var)    