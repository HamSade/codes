#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 15:20:21 2017
@author: hsadeghi
Multi-feature conv. Disc and Generator
"""

import tensorflow as tf
import numpy as np

from ops_codec import mf_downconv, mf_nn_deconv, conv1d
from ops_codec import prelu, leakyrelu#, nn_deconv
from ops_codec import SEGAN

from tensorflow.contrib.layers import fully_connected,flatten

#%% Parameters
filter_width = 31
apply_BN = True
apply_skips = True

#%% In this first version I will make it similar to the encoder
segan = SEGAN()

def discriminator(x, comp_ratio):
#    input_dim = x.get_shape().as_list()[-1] #x.shape[-1]
    num_filters = int(np.log2(comp_ratio))       
    h = x;
    with tf.variable_scope('disc'):
#        if apply_BN:
#            h = segan.vbn(h, 'disc_batch_norm_{}'.format(0))
        for i in range(num_filters):
            print('disc_layer_{} = '.format( i))
            h = mf_downconv(h, output_dim = 2 ** (i + 1) , stride = 2,
                            filter_width=filter_width, name='disc_downconv_{}'.format(i)) 
            print('h_dim= ', h.get_shape().as_list())
            if apply_BN:
                h = segan.vbn( h, 'disc_vbn_{}'.format(i))
            h = leakyrelu(h)   
        # Last layer
        h = flatten(h)
        h_logit_out = conv1d(h, filter_width=1, output_dim=1,
                             w_init = tf.truncated_normal_initializer(stddev=0.02),
                             name='disc_logits_conv')  # 1 x 1 convolution
        d_logit_out = tf.squeeze(h_logit_out) # Remove dimension of 1 comming from conv1d
        disc_output = fully_connected(d_logit_out, 1, activation_fn = None) #tf.tanh)
        return disc_output

#%% Autoencoder
def auto_encoder(x, comp_ratio, noise_std): 
    h, skips = encoder(x, comp_ratio, noise_std)
    h_q = h  # No quantization
    x_ = decoder(h_q, skips, comp_ratio)  
    return x_

#%% 
def encoder(x, comp_ratio, noise_std, activation='prelu'):     
#    input_dim = x.shape[-1] #x.get_shape().as_list()[-1]
    num_filters = int(np.log2(comp_ratio))       
    h = x;
    if len(x.get_shape().as_list())<3:
        h = tf.expand_dims(h, axis =2)
    skips = []
    for i in range(num_filters):
        print('enc_layer_{}'.format( i))
        with tf.variable_scope('enc'):
            h = mf_downconv(h, output_dim = 2**(i + 1) , stride = 2,
                         filter_width=filter_width, name='downconv_{}'.format(i)) 
            print('h_dim= ', h.get_shape().as_list())
            # Skip connections
            if i < num_filters - 1:
                skips.append(h)
                print('skip_dim= ', h.get_shape().as_list())
            if apply_BN:
                h = segan.vbn( h, 'ae_enc_vbn_{}'.format(i))
            if activation=='leakyrelu':
                h = leakyrelu(h)
            else:
                with tf.variable_scope('enc'):
                    h , _ = prelu(h, name='prelu_{}'.format(i))
    z = make_z(h.get_shape().as_list(), std = noise_std)
#    h = h + z
    h = tf.concat([z, h], 2)  
    print('h_and_z_dim', h.get_shape().as_list())                
    return h, skips

#%%
def make_z(shape, std, mean=0., name='z'):
    z = tf.random_normal(shape, mean=mean, stddev=std,
                         name=name, dtype=tf.float32)
    return z
#%% There is no compression actaly! :D Filter features double as frame size shrinks 
def decoder(x, skips, comp_ratio,  activation='leakyrelu'):    
#    input_dim = x.shape[-1] #tf.shape(x)[-1]
    num_filters = int(np.log2(comp_ratio))     
    h = x;
    for i in range(num_filters):
        print('dec_layer_{} '.format( i))   
        with tf.variable_scope('dec'):
            print('h_dim= ', h.get_shape().as_list())
            if apply_skips:
                if i > 0:
                    h = tf.concat([h, skips.pop()], 2)
                    print('h_and_skip_dim= ', h.get_shape().as_list())
            h = mf_nn_deconv(h, output_dim = 2**(num_filters-i-1) , filter_width=filter_width, name='nn_deconv_{}'.format(i),
                  dilation=2, init=tf.truncated_normal_initializer(stddev=0.02))
            if apply_BN:
                h = segan.vbn( h, 'ae_dec_vbn_{}'.format(i))
            if activation=='leakyrelu':
                h = leakyrelu(h)
            else:
                with tf.variable_scope('dec'):
                    h,_ = prelu(h, name='prelu_{}'.format(i))
    # making h of size [num_batch, input_dim]
    h = tf.squeeze(h, axis=-1)
    return h