#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 19:50:57 2017

@author: hsadeghi
"""
#%%
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from bnorm_segan import VBN

#%%
class SEGAN():
    disable_vbn =  False

    def vbn(self, tensor, name):
            if self.disable_vbn:
                class Dummy(object):
                    # Do nothing here, no bnorm
                    def __init__(self, tensor, ignored):
                        self.reference_output=tensor
                    def __call__(self, x):
                        return x
                VBN_cls = Dummy
            else:
                VBN_cls = VBN
            if not hasattr(self, name):
                vbn = VBN_cls(tensor, name)
                setattr(self, name, vbn)
                return vbn.reference_output
            vbn = getattr(self, name)
            return vbn(tensor)

#%% Single feature
def downconv(x, filter_width=31, stride=2, output_channels=1, name=None,  uniform=False,
             w_init = tf.truncated_normal_initializer(stddev=0.02), 
             bias_init=tf.constant_initializer([0.])):
    ''' Downsampled convolution 1d '''
    x2d = tf.expand_dims(x, -1) 
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)    
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width, 1, output_channels], initializer=w_init)
        b = tf.get_variable('b', [1], initializer=bias_init)
    conv= tf.nn.conv1d(x2d, W, stride=stride, padding='SAME')       
    conv = tf.nn.bias_add(conv, b)
    conv = tf.squeeze(conv)   
    return conv

#%% Multi-feature filter
def mf_downconv(x, output_dim, filter_width=31, stride=2, output_channels=1, name=None,  uniform=False,
             w_init = tf.truncated_normal_initializer(stddev=0.02), 
             bias_init=tf.constant_initializer([0.])):
    if len(x.get_shape().as_list())<3:
        x = tf.expand_dims(x, 2)
    x2d = tf.expand_dims(x, 2)
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)    
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width, 1, x.get_shape()[-1], output_dim], initializer=w_init)
        b = tf.get_variable('b', [output_dim], initializer=bias_init)      
    conv = tf.nn.conv2d(x2d, W, strides=[1, stride, 1, 1], padding='SAME')  
    conv = tf.nn.bias_add(conv, b)
#    conv = tf.squeeze(conv) 
    conv = tf.reshape(conv, conv.get_shape().as_list()[:2] +
                          [conv.get_shape().as_list()[-1]])  
    return conv

#%%
def mf_nn_deconv(x, output_dim, filter_width=31, dilation=2, init=None, uniform=False,
              bias_init=None, name='nn_deconv'):
    interp_x = repeat_elements(x, dilation, 1)
    dec = mf_conv1d(interp_x, filter_width=31, output_dim=output_dim,
                 w_init=init, uniform=uniform, 
                 bias_init=bias_init, name=name, padding='SAME')
    return dec

#%%
def mf_conv1d(x, filter_width=31, output_dim=1, w_init=None,
              uniform=False, bias_init=tf.constant_initializer([0.]),
              name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
#    x2d = tf.expand_dims(x, 2)   
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width,in_channels, output_dim], initializer=w_init)
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        b = tf.get_variable('b', [output_dim], initializer=bias_init)
        conv = tf.nn.bias_add(conv, b)
#        conv = tf.squeeze(conv) 
        return conv

#%%
def repeat_elements(x, rep, axis):
    shape = x.get_shape().as_list()  #    shape = tf.shape(x)
    splits = tf.split(x, axis=axis, num_or_size_splits=shape[axis])
    x_rep = [s for s in splits for _ in range(rep)]
    return tf.concat(x_rep, axis)
     
#%%
#x = tf.random_normal([128,2**9, 2])
#y = mf_downconv(x, 4)
#x_ = mf_nn_deconv(y,2)

#init = tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)

#print(sess.run(y))
#print(y.get_shape().as_list())
#print(x_.get_shape().as_list())

#%%
def gated_deconv(x, filter_width=31, dilation=2, init=None, uniform=False,
              bias_init=None, name='gated_deconv'):
    interp_x = repeat_elements(x, dilation, 1)
    dec_tanh = conv1d(interp_x, filter_width=31, w_init=init, uniform=uniform, 
                 bias_init=bias_init, name=name+'_tanh', padding='SAME')
    dec_sigmoid = conv1d(interp_x, filter_width=31, w_init=init, uniform=uniform, 
                 bias_init=bias_init, name=name+'_sigmoid', padding='SAME')
    dec =  tf.multiply(tf.tanh(dec_tanh), tf.sigmoid(dec_sigmoid))
    return dec

#%%
def nn_deconv(x, filter_width=31, dilation=2, output_channels=1, init=None, uniform=False,
              bias_init=None, name='nn_deconv'):
    interp_x = repeat_elements(x, dilation, 1)
    dec = conv1d(interp_x, filter_width=31, w_init=init, uniform=uniform, 
                 bias_init=bias_init, name=name, padding='SAME')
    return dec

#%%
def conv1d(x, filter_width=31, w_init=None, output_dim=1, uniform=False, bias_init=tf.constant_initializer([0.]),
           name='conv1d', padding='SAME'):

    if len(x.get_shape().as_list())<3:
        x2d = tf.expand_dims(x, 2)
    else:
        x2d = x
    if w_init is None:
        w_init = xavier_initializer(uniform=uniform)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [filter_width, 1, output_dim], initializer=w_init)
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

def batch_norm(x, scope):
    num_neurons=x.get_shape()[-1].value;
    epsilon=1e-5;
    x_BN = x
    batch_mean, batch_var = tf.nn.moments(x_BN,[0])  
    
    try:
        with tf.variable_scope(scope):      
            scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
            beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)  
    except ValueError:
        with tf.variable_scope(scope, reuse=True):
            scale = tf.get_variable('scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
            beta  = tf.get_variable('beta', shape=[num_neurons] ,dtype=tf.float32)
            
    x_BN = tf.nn.batch_normalization(x_BN,batch_mean,batch_var,beta,scale,epsilon)
    return x_BN

#%%
def batch_norm_w(x, W, scope):    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])          
    scale = tf.get_variable(scope +'_scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
    beta  = tf.get_variable(scope +'_beta', shape=[num_neurons] ,dtype=tf.float32)  
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)
    return x_BN

#%%
def w_b_gen(shape_param, stddev_param):
    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
    return weight

