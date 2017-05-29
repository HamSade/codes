#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 19:50:57 2017

@author: hsadeghi
"""
#%%
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

#%% Output_dim should be one, because our filter only extracts one feature
def downconv(x, filter_width=3, stride=2, name=None,  uniform=False,
             w_init = tf.truncated_normal_initializer(stddev=0.02), 
             bias_init=tf.constant_initializer([0.])):
    ''' Downsampled convolution 1d '''
    x2d = tf.expand_dims(x, -1) 
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)    
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width, 1, 1], initializer=w_init)
        b = tf.get_variable('b', [1], initializer=bias_init)
    conv= tf.nn.conv1d(x2d, W, stride=stride, padding='SAME')       
    conv = tf.nn.bias_add(conv, b)
    conv = tf.squeeze(conv)   
    return conv

#%%
def nn_deconv(x, filter_width=31, dilation=2, init=None, uniform=False,
              bias_init=None, name='nn_deconv'):
    interp_x = repeat_elements(x, dilation, 1)
    dec = conv1d(interp_x, filter_width=31, w_init=init, uniform=uniform, 
                 bias_init=bias_init, name=name, padding='SAME')
    return dec

#%%
def repeat_elements(x, rep, axis):
    shape = x.get_shape().as_list()  #    shape = tf.shape(x)
    splits = tf.split(x, axis=axis, num_or_size_splits=shape[axis])
    x_rep = [s for s in splits for _ in range(rep)]
    return tf.concat(x_rep, axis)

#%%
def conv1d(x, filter_width=31, w_init=None, uniform=False, bias_init=tf.constant_initializer([0.]),
           name='conv1d', padding='SAME'):

    x2d = tf.expand_dims(x, 2)
    
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width, 1, 1], initializer=w_init)
        conv = tf.nn.conv1d(x2d, W, stride=1, padding=padding)
        b = tf.get_variable('b', [1], initializer=bias_init)
        conv = tf.nn.bias_add(conv, b)
        conv = tf.squeeze(conv) 
        return conv

#%%
def prelu(x, name='prelu'):
    alpha_size = x.get_shape()[-1]
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', alpha_size,initializer=tf.constant_initializer(0.),
                                dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        return pos + neg, alpha
    
#%%
def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x)

#%% Full layer

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
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])          
    scale = tf.get_variable(scope +'scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
    beta  = tf.get_variable(scope +'beta', shape=[num_neurons] ,dtype=tf.float32)  
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)
    return x_BN

#%%
def w_b_gen(shape_param, stddev_param):
    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
    return weight


