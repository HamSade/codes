#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 20:57:07 2017

@author: hsadeghi
"""

#%%
import tensorflow as tf
import numpy as np

#%%
depths_disc= [1, 64, 128, 256, 512]
depths_gen =   [1024, 512, 256, 128, 1] 

#%%
def generator(x_shape, z_dim= 100, training = 0., activation = tf.tanh):
    training_ind = training > 0.
    batch_size = x_shape[0]
#    y = tf.convert_to_tensor(x)
#    y = tf.expand_dims(y, axis=-1)  # to indicate #channels = 1
    z = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0)
#    z = tf.random_normal(shape= [batch_size, z_dim], stddev= 1)
    h_size = x_shape[1] // 16 #depends on nfft
    w_size = x_shape[2] // (2** 11)  #depends on nperseg
    print('initial rectangle size', [h_size, w_size])
    with tf.variable_scope('reshape'):
        y = tf.layers.dense(z, depths_gen[0] * h_size * w_size)
        y = tf.reshape(y, [-1, h_size, w_size, depths_gen[0]])
        y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_reshape')

    with tf.variable_scope('gen'):
        for i in range(1, len(depths_gen)):
            with tf.variable_scope('deconv_{}'.format(i)):
                print('Deconv layer', i)
                y = tf.layers.conv2d_transpose(y, depths_gen[i], [3, 3], strides=(2, 2), padding='SAME')
                if i < len(depths_gen) - 1:
                    y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs')
        if activation:
            with tf.variable_scope('disc_activation'):
                y = activation(y, name='outputs_tanh')
                
        return tf.squeeze(y)
            
#%%        
def leaky_relu(x, leak=0.2, name='leakyrelu'):
            return tf.maximum(x, x * leak, name=name)
        
#%%
def discriminator(x, training = 0., activation= leaky_relu):
    
    training_ind = training > 0.
    y = tf.convert_to_tensor(x)
    if len(x.get_shape().as_list()) < 4:
        y = tf.expand_dims(y, axis=-1)  # to add dimension 1 to our inout which is 3 dimensional
    with tf.name_scope('disc'):
        for i in range(1, len(depths_disc)):
            with tf.variable_scope('downconv_{}'.format(i)):
                y = tf.layers.conv2d(y, depths_disc[i], [3, 3], strides=(2, 2), padding='SAME')
                y = activation(tf.layers.batch_normalization(y, training=training_ind), name='outputs_leakyrelu')   
    with tf.variable_scope('classify'):
        batch_size = y.get_shape()[0].value
        reshape = tf.reshape(y, [batch_size, -1, 1]) #tf.reshape(outputs, [batch_size, -1])
        y = tf.layers.dense(reshape, 1, activation= tf.nn.relu, name='outputs_dense')
    return y

#%% test disc
#x = tf.random_normal(shape=[1, 256, 16])
#y = discriminator(x)
#sess=tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#y = sess.run(y)
#print(y)