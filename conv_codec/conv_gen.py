#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:40:15 2017

@author: hsadeghi
"""
import tensorflow as tf
import numpy as np

from ops_codec import downconv, prelu, leakyrelu, nn_deconv
from ops_codec import full_layer, gated_deconv, batch_norm

#from ops_codec import batch_norm, full_layer, w_b_gen

from binary_quantizer import binary_quantizer

#%% Parameters
#num_filters = 7  #has to be determined based on compression ratio

filter_width = 31
apply_BN = False


#%%
#def discriminator(x):    
#    x_ = encoder(x, activation='leakyrelu')
#    full_width = ...
#    return o

#%% Generator
def auto_encoder(x, mode, comp_ratio=4): 
    if comp_ratio==0:
        return tf.zeros(x.get_shape().as_list()), tf.zeros([x.get_shape().as_list()[0],1]) 
    else:        
        h = encoder(x, comp_ratio)
#        h = tf.tanh(h)

        shape = h.get_shape().as_list()
        h_ = tf.reshape(h, [shape[0], shape[1], -1])
        h_ = tf.abs(h_)
        eps = 10**-8
        h_max = tf.reduce_max(h_, axis=-1)
        h = tf.divide(h, h_max +  eps)
        
        h_q = binary_quantizer(h, mode)
        codes = h_q
        x_ = decoder(h_q, comp_ratio)  
        return x_, codes
    
#%% Assuming x is batch_size x input_dim
def encoder(x, comp_ratio, activation='prelu'):     
    input_dim = x.shape[-1] #tf.shape(x)[-1]
    num_filters = int(np.log2(comp_ratio))         
    h = x; 
    if apply_BN:
        h = batch_norm(h, 'enc_batch_norm_{}'.format(0))
     
    for i in range(num_filters):
        print('enc_layer_number = ', i)
        print('h_dim= ', h.get_shape().as_list())
        
        with tf.variable_scope('g_enc'):
            h = downconv(h, filter_width=filter_width, name='downconv_{}'.format(i)) 
            if apply_BN:
                h = batch_norm(h, 'batch_norm_{}'.format(i))
        if activation=='leakyrelu':
            h = leakyrelu(h)
        else:
            with tf.variable_scope('g_enc'):
                h , _ = prelu(h, name='prelu_{}'.format(i))                  
#    with tf.variable_scope('g_enc'):
#        w_full_enc = tf.get_variable('w_full_enc', [ input_dim / (2**num_filters), input_dim / comp_ratio],
#                                  initializer= tf.random_normal_initializer(stddev=0.01))
#        h = full_layer(h, w_full_enc, 'g_enc')        
    return h

#%%
def decoder(x, comp_ratio,  activation='leakyrelu', nonlin='segan'): #gated_conv'):    
    input_dim = x.shape[-1] #tf.shape(x)[-1]
    num_filters = int(np.log2(comp_ratio))     
    h=x;
    if apply_BN:
        h = batch_norm (h, 'dec_batch_norm_{}'.format(0))
#    with tf.variable_scope('g_dec'):
#        w_full_dec = tf.get_variable('w_full_dec', [input_dim ,
#                                                    input_dim * comp_ratio / (2**num_filters) + 1],
#                                  initializer= tf.random_normal_initializer(stddev=0.02))  
#        h = full_layer(x, w_full_dec, 'g_dec')
    for i in range(num_filters):
        print('dec_layer_number = ', i)
        print('h_dim= ', h.get_shape().as_list())
        with tf.variable_scope('g_dec'):
            if nonlin=='segan':
                #original segan
                h = nn_deconv(h, filter_width=filter_width, name='nn_deconv_{}'.format(i),
                      dilation=2, init=tf.truncated_normal_initializer(stddev=0.02)) 
                if apply_BN:
                    h = batch_norm (h, 'batch_norm_{}'.format(i))
                if activation=='leakyrelu':
                    h = leakyrelu(h)
                else:
                    with tf.variable_scope('g_dec'):
                        h,_ = prelu(h, name='prelu_{}'.format(i))
            else:
            # Wavenet gated convolutions
                h = gated_deconv(h, filter_width=filter_width, name='gated_deconv_{}'.format(i),
                          dilation=2, init=tf.truncated_normal_initializer(stddev=0.02))
    return h