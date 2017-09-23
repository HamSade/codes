#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:36:20 2017
@author: hsadeghi
Multi-feature conv layers
"""
import tensorflow as tf
import numpy as np

from ops_codec import mf_downconv, mf_nn_deconv
from ops_codec import prelu, leakyrelu, nn_deconv
from ops_codec import full_layer, gated_deconv, batch_norm
#from ops_codec import batch_norm, full_layer, w_b_gen
from binary_quantizer import binary_quantizer

#%% Parameters

filter_width = 31
apply_BN = True

#%% Generator
def auto_encoder(x, mode, comp_ratio=4): 
    if comp_ratio==0:
        return tf.zeros(x.get_shape().as_list()), tf.zeros([x.get_shape().as_list()[0],1]) 
    else:        
        h = encoder(x, comp_ratio)
#        h = tf.tanh(h)
        
#        shape = h.get_shape().as_list()
#        h_ = tf.reshape(h, [shape[0], shape[1], shape[2],-1])
#        h_ = tf.abs(h_)
#        eps = 10**-8
#        h_max = tf.reduce_max(h_, axis=-1)
#        h = tf.divide(h, h_max +  eps)
        
        eps = 10**-8
        h_ = tf.abs(h)
        h_max = tf.reduce_max(h_)
        h = tf.divide(h, h_max+eps)

        h_q = binary_quantizer(h, mode)
        x_ = decoder(h_q, comp_ratio)  
        return x_, h_q #codes = h_q
    
#%%
#def discriminator(x):    
#    x_ = encoder(x, activation='leakyrelu')
    #full_width = ...
    #    with tf.variable_scope('g_enc'):
#        w_full_enc = tf.get_variable('w_full_enc', [ input_dim / (2**num_filters), input_dim / comp_ratio],
#                                  initializer= tf.random_normal_initializer(stddev=0.01))
#        h = full_layer(h, w_full_enc, 'g_enc')  
#    return o
#%% Assuming x is batch_size x input_dim
def encoder(x, comp_ratio, activation='prelu'):     
    input_dim = x.shape[-1] #x.get_shape().as_list()[-1]
    num_filters = int(np.log2(comp_ratio))       
    h = x;
    if apply_BN:
        h = batch_norm(h, 'enc_batch_norm_{}'.format(0))
    for i in range(num_filters):
        print('enc_layer_number = ', i)
        print('h_dim= ', h.get_shape().as_list())
        with tf.variable_scope('g_enc'):
            h = mf_downconv(h, output_dim = 2**i , stride = 4,
                         filter_width=filter_width, name='downconv_{}'.format(i)) 
            if apply_BN:
                h = batch_norm(h, 'batch_norm_{}'.format(i))
        if activation=='leakyrelu':
            h = leakyrelu(h)
        else:
            with tf.variable_scope('g_enc'):
                h , _ = prelu(h, name='prelu_{}'.format(i))                   
    return h
#%%
def decoder(x, comp_ratio,  activation='leakyrelu', nonlin='segan'): #gated_conv'):    
    input_dim = x.shape[-1] #tf.shape(x)[-1]
    num_filters = int(np.log2(comp_ratio))     
    h = x;
    if apply_BN:
        h = batch_norm (h, 'dec_batch_norm_{}'.format(0))
    for i in range(num_filters):
        print('dec_layer_number = ', i)   
        print('h_dim= ', h.get_shape().as_list())
        with tf.variable_scope('g_dec'):
            if nonlin=='segan':
                #original segan
                h = mf_nn_deconv(h, output_dim = 2**(num_filters-i-1) , filter_width=filter_width, name='nn_deconv_{}'.format(i),
                      dilation=4, init=tf.truncated_normal_initializer(stddev=0.02)) 
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
    
    # making h of size [num_batch, input_dim]
    h = tf.squeeze(h, axis=-1)
    return h

#%% checking generator
#x= tf.random_normal([2, 2**14])
#y= auto_encoder(x,1.)
#init = tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
##print(sess.run(y))
#print(sess.run(tf.shape(y)))

#%% checking encoder/decoder
#x= np.random.normal(size=[2, 2**11]).astype(np.float32)
#x= tf.random_normal([2, 2**11])
#y= decoder(x)
#init = tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
##print(sess.run(y))
#print(sess.run(tf.shape(y)))
#%% checking nn_deconv
#x=tf.random_uniform([3,5])
#y=nn_deconv(x, filter_width=2)
#init = tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#print('x=',sess.run(x))
#print('y=', sess.run(y))
#print('y_shape', sess.run(tf.shape(y)))
#%% testing downconv
#x=[[0,1,2,3,4,5]]
#x=np.array(x).astype(np.float32)
##phi=[1,0,1]
#y=  downconv(x, filter_width=3, stride=2)
#z= downconv([y], filter_width=3, stride=1)
#init = tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)2000
#print(sess.run(y))
#print(sess.run(z))
#%% testing repeat_elements
#x=tf.random_uniform([2,3])
##y=tf.tile(x, [1,2])
#y=repeat_elements(x, 2, axis=1)
#sess=tf.Session()
#print(sess.run([x,y]))
#%% 

