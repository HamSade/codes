#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:19:08 2017

@author: hsadeghi
"""
import tensorflow as tf


def full_layer(x, w, scope):
    
    try:
        with tf.variable_scope(scope, reuse=True):
            tf.get_variable_scope().reuse_variables()
            layer = batch_norm(x, w, scope)
            print('reuse')
    except ValueError:
        with tf.variable_scope(scope):
            print('creation')
            layer = batch_norm(x, w, scope)
#        layer = tf.matmul(x, w)
    layer = tf.nn.tanh(layer);  
    
    return layer

#%%
def batch_norm(x, W, scope):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-3;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
          
    scale = tf.get_variable(scope +'scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
    beta  = tf.get_variable(scope +'beta', shape=[num_neurons] ,dtype=tf.float32)  

    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN