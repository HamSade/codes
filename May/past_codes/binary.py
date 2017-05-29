#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:22:56 2017

@author: hsadeghi
"""

#%%

from __future__ import division

import tensorflow as tf


#%%
def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
#    with tf.name_scope('encode'):
    mu = tf.to_float(quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.to_int32((signal + 1) / 2 * mu + 0.5)

#%%
def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
#    with tf.name_scope('decode'):
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (tf.to_float(output) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude
    
#%% 
def float_to_one_hot(x, n):

    x_compressed = mu_law_encode(x, 2**n)

    x_1_hot = tf.one_hot(x_compressed, 2**n)  
    
    return x_1_hot

#%%
def one_hot_to_float(x, n):

    x_int = tf.argmax(x, axis=1)
    
    x_float= mu_law_decode(x_int, 2**n)  
    
    return x_float

#%%

#
#from binary_quantizer import binary_quantizer
#
#
##%%
#
#def bitwise_layer(x, width, W, b, tf, mode):
#    
#    # W must beof size [width+1, width]
#    
##    b_q = binary_quantizer(tf)  #binary quantizer initialization
#    
#    
##    out_temp = tf.zeros([ tf.shape(x)[0], width]);
#    
##    xo =x;  # considering x is batch_size by 1
#    
#    out_temp = tf.matmul ( x, W[0]) + b[0]
#    
#    xo = tf.concat([x, out_temp], axis=1)
#    
#    for i in range(width-1):
#        
#        out_temp = tf.concat([out_temp,  tf.matmul( xo, W[i+1])+ b[i+1] ], axis=1)    
#
##        out_temp = tf.concat([out_temp, out_temp], axis=1)
#
#        xo = tf.concat([x, out_temp], axis=1)
#    
#                
#        

#    output = tf.stack(out_temp);
#    
#    return output
##    return b_q (output, mode)
#

#%%

#def bitwise_layer_2(x, width, W, b, tf, mode):
#    
#    # W must beof size [width+1, width]
#    
##    b_q = binary_quantizer(tf)  #binary quantizer initialization
#    
#    
##    out_temp = tf.zeros([ tf.shape(x)[0], width]);
#    
##    xo =x;  # considering x is batch_size by 1
#    
#    out_temp = tf.matmul ( x, W[0]) + b[0]
#    
#    xo = tf.concat([x, out_temp], axis=1)
#    
#    for i in range(width-1):
#        
#        out_temp = tf.concat([out_temp,  tf.matmul( xo, W[i+1])+ b[i+1] ], axis=1)    
#
##        out_temp = tf.concat([out_temp, out_temp], axis=1)
#
#        xo = tf.concat([x, tf.reshape( out_temp[:,-1] , [-1, 1] ) ], axis=1)
#    
#                
#        
#
#    output = tf.stack(out_temp);
#    
#    return output
#
#
#
#
##%%
#def binary_cost(o, t, tf):
#    
#    return 0.5 * tf.reduce_sum( 1. - tf.multiply(o, t), 1)
#    
##%%

# float to n-bit binary format conversion for inputs in [-1,1]

#def dec_to_bin(x, n, tf):
#    
#    x_2 = x;
#    x_b = tf.zeros([tf.shape(x)[0], 1])
#    
#    for _ in range(n):
#        
#        x_2 = tf.cast( tf.cast( 2 * x_2 , tf.int32), tf.float32)  
#
#        x_b = tf.concat ( [x_b, x_2], axis=1) 
#        
##        x_2 = 
#
#    return x_b[:, 1:]




#%% #################################
#########   Testing functions   ######
#####################################

#b=dec_to_bin(0.99*tf.ones([2,1]), 4, tf); sess.run(b)

#%% Testing functions

#import tensorflow as tf
#
## x samples are in [-1, 1]
#x=2*tf.random_uniform([4,1])-1
#width=3; # length of binary represenattion
#
#
#W=[]
##tf.random_normal([1], stddev=0.01)
#
#for i in range(width):
#    W.append(tf.random_normal([i+1,1], stddev=0.01))
#
#b= tf.random_normal([width+1], stddev=0.001)
#
#
#
#
#y= bitwise_layer(x, width, W,b,  tf, 1)
#
#
#sess=tf.Session()
#
#sess.run([x,y])












