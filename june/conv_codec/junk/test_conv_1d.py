#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 20:04:16 2017

@author: hsadeghi
"""

import tensorflow as tf



x=[[0.,1.,2.,3.,4.,5.]]
x3d=tf.expand_dims(x, axis=2)

phi=[1., 0., 1.]
phi2d=tf.expand_dims(phi,1)
phi3d=tf.expand_dims(phi2d,2)

y= tf.nn.conv1d(x3d, phi3d, stride=2, padding='SAME')

y_b = tf.nn.bias_add(y, [1.])


sess=tf.Session()


print('x = ', sess.run(x3d))

print('phi_2d = ', sess.run(phi2d))
print('phi_3d = ', sess.run(phi3d))


print(sess.run(tf.shape(x3d)[0]))
print(sess.run(tf.shape(x3d)[1]))
print(sess.run(tf.shape(x3d)[2]))

print('y = ', sess.run(y))
print('y_b = ', sess.run(y_b))

print('y = ', sess.run(tf.shape(y)))
print('y_b = ', sess.run(tf.shape(y_b)))

print('squeezed_y_b = ', sess.run(tf.squeeze(y_b)))
