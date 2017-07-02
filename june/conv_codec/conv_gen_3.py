#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:52:08 2017

@author: hsadeghi

Refinement info (side_net) is also convolutional
"""

import tensorflow as tf
import numpy as np

from ops_codec import downconv, prelu, leakyrelu, nn_deconv
from ops_codec import full_layer, gated_deconv, batch_norm

#from ops_codec import batch_norm, full_layer, w_b_gen

from binary_quantizer import binary_quantizer

#%% Parameters
#num_filters = 7  #has to be determined based on compression ratio

comp_ratio = 4
filter_width = 31

#%% Generator
def auto_encoder(x, mode):  
    input_dim = x.shape[-1]
    h, hs = encoder(x)
    h_q = binary_quantizer(h, mode)
    ps_q = side_net(hs, mode)
    x_dec = decoder(h_q)
    x_ = mux(x_dec, ps_q, input_dim)
    return x_

#%%
def mux(x_dec, ps_q, input_dim):
    x = x_dec #coming from decoder
    for i in range(len(ps_q)):
        x = tf.concat([x, ps_q[i]], axis=1)  #coming from encoder
    total_code_length = x.get_shape().as_list()[-1] 
    print('total_code_length', total_code_length)
    w_mux = tf.get_variable('w_mux', [total_code_length, input_dim],
                                                    initializer= tf.random_normal_initializer(stddev=0.01))
    x_ = full_layer(x, w_mux, 'g_mux') 
    return x_

#%% refinement info
def side_net(hs, mode ):
#    input_dim = hs[0].get_shape().as_list()[-1]
    num_filters = len(hs)
#    code_lengths = input_dim/50 * np.power(2, range(num_filters,0,-1))
#    print('code_lengths', code_lengths)
    ps_q=[]
    for i in range(len(hs)):
        with tf.variable_scope('g_side_' + str(i) ):
            h_temp = downconv(hs[i], filter_width=filter_width/(2**(num_filters - i)), stride=2**(comp_ratio - i),
                         name='downconv')
#            h_temp = downconv(hs[i], filter_width=filter_width, stride=2,
#                         name='downconv')  
            h_temp = batch_norm(h_temp, 'batch_norm_{}'.format(i))
            ps_q.append( binary_quantizer(h_temp, mode) ) 
    return ps_q

#%% Assuming x is batch_size x input_dim
def encoder(x, activation='prelu'):     
    num_filters = int(np.log2(comp_ratio))         
    h = x;
    hs = []     
    for i in range(num_filters):
        print('enc_layer_number = ', i)
        hs.append(h)
        with tf.variable_scope('g_enc'):
            h = downconv(h, filter_width=filter_width, name='downconv_{}'.format(i)) 
            h = batch_norm(h, 'batch_norm_{}'.format(i))
        if activation=='leakyrelu':
            h = leakyrelu(h)
        else:
            with tf.variable_scope('g_enc'):
                h , _ = prelu(h, name='prelu_{}'.format(i))                   
    return h, hs

#%%
def decoder(x, activation='leakyrelu', nonlin='segan'): #gated_conv'):    
#    input_dim = x.shape[-1]
    num_filters = int(np.log2(comp_ratio))     
    h=x;
    for i in range(num_filters):
        print('dec_layer_number = ', i)       
        with tf.variable_scope('g_dec'):
            if nonlin=='segan':
                #original segan
                h = nn_deconv(h, filter_width=filter_width, name='nn_deconv_{}'.format(i),
                      dilation=2, init=tf.truncated_normal_initializer(stddev=0.02)) 
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