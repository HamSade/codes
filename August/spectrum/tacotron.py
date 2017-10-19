#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:33:00 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np

from ops_codec import mf_downconv  #, mf_nn_deconv, conv1d
from ops_codec import prelu, leakyrelu#, nn_deconv
from ops_codec import SEGAN

from highway_ops import highway_layer_w, deep_highway_with_carrier, full_layer
#from tensorflow.contrib.layers import fully_connected,flatten

#%% Parameters
filter_width = 31
apply_BN = True
apply_skips = True

#%%
def taco_BWE(x, num_conv, num_highway):
    
    signal = x[0]
    noise = x[1]
    x_total = tf.concat([signal, noise], axis = -1)
    
    #conv
    y = conv_net(x_total, num_layers = num_conv)
#    W =  tf.Variable(tf.truncated_normal([x_total.get_shape()[-1].value, y.get_shape()[-1].value,
#                                          stddev=0.1), name="weight_res")
    y = tf.subtract( y , x_total)
    
    #highway
    signal, noise = tf.split(y, 2, axis = 1)
    x_in_highway = (signal, noise)
    input_dim = x_total.get_shape().as_list()[-1]
    highway_widths = [input_dim//2] * num_highway
#    highway_widths = np.arange(input_dim, input_dim//2 , -int(input_dim/2/(num_highway-1)))
    y = deep_highway_with_carrier(x_in_highway , highway_widths)
    return y

#%% First highway disc ever!!!!!! :D
def highway_disc(x, num_layers):
    
    activation = tf.nn.relu
    input_dim = x.get_shape().as_list()[-1]
    # Triangle-shaped net to classify fake/real
    layers = np.arange(input_dim, int(input_dim/10), -int(9 * input_dim/10/ (num_layers-1))) #10 layers    
    h = x
    for i, lw in enumerate(layers):
        with tf.variable_scope('highway_layer_env_{}'.format(i)) as scope:
            h = highway_layer_w(h, lw, activation =activation, scope = scope)
    
#    h = tf.squeeze(h) # Remove dimension of 1 comming from conv1d
    h = highway_layer_w(h, 1, activation = tf.nn.relu, scope = 'disc_out')
    return h

#%%   
segan = SEGAN()

def conv_net(x, num_layers, activation='prelu'):     
#    input_dim = x.shape[-1] #x.get_shape().as_list()[-1]      
    h = x;
    if len(x.get_shape().as_list())<3:
        h = tf.expand_dims(h, axis =2)
#    skips = []
    for i in range(num_layers):
        print('conv_layer_{}'.format( i))
        with tf.variable_scope('conv'):
            h = mf_downconv(h, output_dim = 1 , stride = 1,
                         filter_width = 31, name='downconv_{}'.format(i)) 
            print('h_dim= ', h.get_shape().as_list())
            # Skip connections
#            if i < num_filters - 1:
#                skips.append(h)
#                print('skip_dim= ', h.get_shape().as_list())
            if apply_BN:
                h = segan.vbn( h, 'ae_enc_vbn_{}'.format(i))
            if activation=='leakyrelu':
                h = leakyrelu(h)
            else:
                with tf.variable_scope('enc'):
                    h , _ = prelu(h, name='prelu_{}'.format(i))
                    
    h = tf.squeeze(h)
    return h #, skips

#%% Test conv layers
#x = tf.zeros(shape=[128, 512])
#y = highway_disc(x, 3)
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#print(sess.run(tf.shape(y)))

#%% Test conv layers
#x = tf.zeros(shape=[128, 512])
#y = conv_net(x, 3)
#sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#print(sess.run(tf.shape(y)))

#%%
#def conv_disc(x, comp_ratio):
##    input_dim = x.get_shape().as_list()[-1] #x.shape[-1]
#    num_filters = int(np.log2(comp_ratio))       
#    h = x;
#    with tf.variable_scope('disc'):
##        if apply_BN:
##            h = segan.vbn(h, 'disc_batch_norm_{}'.format(0))
#        for i in range(num_filters):
#            print('disc_layer_{} = '.format( i))
#            h = mf_downconv(h, output_dim = 2 ** (i + 1) , stride = 2,
#                            filter_width=filter_width, name='disc_downconv_{}'.format(i)) 
#            print('h_dim= ', h.get_shape().as_list())
#            if apply_BN:
#                h = segan.vbn( h, 'disc_vbn_{}'.format(i))
#            h = leakyrelu(h)   
#        # Last layer
#        h = flatten(h)
#        h_logit_out = conv1d(h, filter_width=1, output_dim=1,
#                             w_init = tf.truncated_normal_initializer(stddev=0.02),
#                             name='disc_logits_conv')  # 1 x 1 convolution
#        d_logit_out = tf.squeeze(h_logit_out) # Remove dimension of 1 comming from conv1d
#        disc_output = fully_connected(d_logit_out, 1, activation_fn = None) #tf.tanh)
#        return disc_output
    