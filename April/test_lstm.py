#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 17:13:24 2017

@author: hsadeghi
"""

import tensorflow as tf


lstm = tf.contrib.rnn.BasicLSTMCell(5)
#    state = tf.zeros([batch_size, lstm.state_size])
    

stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 2)


sess=tf.Session()

#a = stacked_lstm.zero_state(3, tf.float32)

a=stacked_lstm.state_size


sess.run(a)

print(a)


#print(a[0][0])
#print(a[0][1])
#
#print(a[1][0])
#print(a[1][1])


sess.close()