#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:40:54 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
#from tensorflow.contrib.layers import fully_connected

#%%
def deep_highway(x, layer_widths):     
    y = x;
    input_dim = y.get_shape()[-1].value;
    # Matching fully connected layer
    W_1 = tf.Variable(tf.truncated_normal([input_dim, layer_widths[0]], stddev=0.1), name="weight")
    y = full_layer(y, W_1, 'full_layer') #, activation=None)
    # highway layers
    for i, lw in enumerate(layer_widths):
        with tf.variable_scope('highway_layer_{}'.format(i)) as scope:
            if i < len(layer_widths) - 1:
                y = highway_layer_w(y, lw, activation = tf.nn.relu, scope = scope)
            else: # last layer
#                y = highway_layer(y, lw, activation = None, scope = scope)
                y = highway_layer_w(y, lw, activation = None, scope = scope)
    return y 

#%%
def highway_layer_w(x, size, activation, scope, carry_bias = -1.0):
    input_dim = x.get_shape()[-1].value; #as_list()[-1]
    W_T = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.01), name="weight_transform")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    W = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.01), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
#    H = batch_norm_w(x, W, scope = scope)
    H = tf.matmul(x , W) + b
    if activation:
        H = activation(H, name="activation")
    #  H = activation(tf.matmul(x, W) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")  # C = 1 - T

    W_x = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.01), name="weight_x")
    b_x = tf.Variable(tf.constant(0.1, shape=[size]), name="b_x")
    
#    x_w = batch_norm_w(x, W_x, scope = 'x_w')
    x_w = tf.matmul(x, W_x) + b_x
    y = tf.add( tf.multiply(H, T), tf.multiply(x_w, C))
    return y
   
#%%
def highway_layer(x, size, activation, scope, carry_bias=-1.0):
    input_dim = x.get_shape()[-1].value; #as_list()[-1]
    W_T = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.01), name="weight_transform")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    W = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.01), name="weight")
    b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = batch_norm_w(x, W, scope = scope)
#    H= tf.matmul(x, W)
    if activation:
        H = activation(H, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")  # C = 1 - T

    y = tf.add( tf.multiply(H, T), tf.multiply(x, C))
    return y

#%%
def batch_norm_w(x, w, scope, training= False): 
    x_BN =  tf.matmul(x, w)    
    x_BN = tf.layers.batch_normalization( x_BN,
                                         axis=-1, epsilon=1e-5, training = training, name= scope,
                                         reuse=None)
    return x_BN

#%%
def full_layer(x, w, scope, activation = tf.nn.relu):   
    with tf.variable_scope(scope):
#        layer = batch_norm_w(x, w, scope)
        layer = tf.matmul(x, w)
        if activation:
            layer = activation(layer);   
        return layer

#%%
#def batch_norm_w(x, W, scope):    
#    num_neurons=W.get_shape()[-1].value;
#    epsilon=1e-5;
#    z_BN = tf.matmul(x , W)   # x * W
#    batch_mean, batch_var = tf.nn.moments(z_BN,[0])          
#    scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
#    beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)  
#    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)
#    return x_BN

#%%
#def full_layer(x, w, scope):   
#    try:
#        with tf.variable_scope(scope, reuse=True):
#            tf.get_variable_scope().reuse_variables()
##            layer = batch_norm_w(x, w, scope)
#            layer = tf.matmul(x, w)
#            print('reuse')
#    except ValueError:
#        with tf.variable_scope(scope):
#            print('creation')
##            layer = batch_norm_w(x, w, scope)
#            layer = tf.matmul(x, w)
#    layer = tf.nn.relu(layer);   
#    return layer
    