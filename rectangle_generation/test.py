#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 22:27:11 2017

@author: hsadeghi
"""

import tensorflow as tf


sess = tf.Session()

#%%
#a= [[1,2,3],[4,5,6]]
#
#b= tf.expand_dims(a, axis=-1)
#
#b = sess.run(b)
#print(b)
#print(b.shape)


#%%
a= [[1,2,3],[4,5,6]]
b= [[10,2,30],[-4,5,-6]]

cost = tf.reduce_mean(tf.squared_difference(a, b))  #confirmed: give a scalar! ;)

print(sess.run(cost))


