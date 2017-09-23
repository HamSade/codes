#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:40:54 2017

@author: hsadeghi
"""

import tensorflow as tf
#from tensorflow.contrib.layers import fully_connected

#%%
def deep_highway(x, layer_widths):     
    activation = tf.tanh
    y = x;
    input_dim = y.get_shape()[-1].value;
    W_1 = tf.Variable(tf.truncated_normal([input_dim, layer_widths[0]], stddev=0.1), name="weight")
    y = full_layer(y, W_1, 'full_layer')
    for i, lw in enumerate(layer_widths):
        with tf.variable_scope('highway_layer_{}'.format(i)) as scope:
            y = highway_layer(y, lw, activation, scope = scope)
    return y 
   
#%%
def deep_highway_with_carrier(x, layer_widths):      
    # x is (x_l, noise)
    signal = x[0]
    noise = x[1]
    x_total = tf.concat([signal, noise], axis = -1)
    
    # dimension matching for the carrier net
    input_dim = x_total.get_shape()[-1].value
    W_1 = tf.Variable(tf.truncated_normal([input_dim, layer_widths[0]], stddev=0.1), name="weight")
    y = full_layer(x_total, W_1, 'full_layer')
#    y = fully_connected(x_total, layer_widths[0], activation_fn = tf.tanh)
    
    # Carrier
    activation = tf.tanh
    y_car = y
    layer_widths_car = [y_car.get_shape()[-1].value] * 10
    for i, lw in enumerate(layer_widths_car):
        with tf.variable_scope('highway_layer_car_{}'.format(i)) as scope:
            y_car = highway_layer(y_car, lw, activation = activation, scope = scope)
    
    # Envelope
    activation = tf.nn.relu
    y_env = signal
    for i, lw in enumerate(layer_widths):
        with tf.variable_scope('highway_layer_env_{}'.format(i)) as scope:
            y_env = highway_layer(y_env, lw, activation =activation, scope = scope)
            
            
    return tf.multiply(y_env, y_car) 
#%%
def highway_layer(x, size, activation, scope, carry_bias=-1.0):
    input_dim = x.get_shape()[-1].value; #as_list()[-1]
    W_T = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.1), name="weight_transform")
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")

    W = tf.Variable(tf.truncated_normal([input_dim, size], stddev=0.1), name="weight")
    #  b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = batch_norm_w(x, W, scope = scope)
    H = activation(H, name="activation")
    #  H = activation(tf.matmul(x, W) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")  # C = 1 - T

    y = tf.add( tf.multiply(H, T), tf.multiply(x, C))
    return y

#%%
def batch_norm_w(x, W, scope):    
    num_neurons=W.get_shape()[-1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x , W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])          
    scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
    beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)  
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)
    return x_BN

#%%
def full_layer(x, w, scope):   
    try:
        with tf.variable_scope(scope, reuse=True):
            tf.get_variable_scope().reuse_variables()
            layer = batch_norm_w(x, w, scope)
            print('reuse')
    except ValueError:
        with tf.variable_scope(scope):
            print('creation')
            layer = batch_norm_w(x, w, scope)
#        layer = tf.matmul(x, w)
    layer = tf.nn.tanh(layer);   
    return layer

#%%
#def highway_layer(x, w, scope):   
#    with tf.variable_scope(scope, reuse=True):
#        tf.get_variable_scope().reuse_variables()
#        out = batch_norm_w(x, w, scope)
##        out = tf.matmul(x, w)
#    out = tf.nn.tanh(out);   
#    return out


#%%

#def batch_norm(x, scope):
#    num_neurons=x.get_shape()[-1].value;
#    epsilon=1e-5;
#    x_BN = x
#    batch_mean, batch_var = tf.nn.moments(x_BN,[0])  
#    
#    try:
#        with tf.variable_scope(scope):      
#            scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
#            beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)  
#    except ValueError:
#        with tf.variable_scope(scope, reuse=True):
#            scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
#            beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)
#            
#    x_BN = tf.nn.batch_normalization(x_BN,batch_mean,batch_var,beta,scale,epsilon)
#    return x_BN

#%%
#def w_b_gen(shape_param, stddev_param):
#    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
#    return weight    