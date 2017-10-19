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
depths_disc= [8, 4, 2, 1]#[8 for _ in range(16)]+[4, 2, 1] #np.power(2, list(range(7,-1,-1))) 
depths_reg =   depths_disc #[4, 2, 1] #np.power(2, list(range(10,-1,-1)), 

#%%
def regressor(x, training = False):
    training_ind = training > 0.
    y = tf.convert_to_tensor(x)
    y = tf.expand_dims(y, axis=-1)  # to indicate #channels = 1
    with tf.variable_scope('reg'):
        for i in range(len(depths_reg)):
            with tf.variable_scope('conv_{}'.format(i)):
                print('Layer', i)
                y = tf.layers.conv2d(y, depths_reg[i], [3, 3], strides=(1, 1), padding='SAME')
                
                if i < (len(depths_reg) - 1):  #last layer is linear (Even without BN!!!)
#                    y = tf.nn.relu(y, name='outputs_relu')
                    y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
#                else:
#                    y = tf.nn.tanh(tf.layers.batch_normalization(y, training=training_ind), name='outputs_tanh')
#                    y = tf.layers.batch_normalization(y, training=training_ind, name='outputs_linear')
        return tf.squeeze(y[:,:-1,:,:])  
    
#%%
def regressor_noise(x, training = False):
    training_ind = training > 0.
    y = tf.convert_to_tensor(x)
    y = tf.expand_dims(y, axis=-1)  # to indicate #channels = 1
    with tf.variable_scope('reg'):
        for i in range(len(depths_reg)):
            with tf.variable_scope('conv_{}'.format(i)):
                if i < (len(depths_reg) - 1):
                    print('Layer', i)
                    y = tf.layers.conv2d(y, depths_reg[i], [3, 3], strides=(1, 1), padding='SAME')
                    y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
#                
                      #last layer is DOWN-convolving
                else:
                    y = tf.layers.conv2d(y, depths_reg[i], [3, 3], strides=(2, 1), padding='SAME', name='outputs_relu_{}'.format(i))
#                    y = tf.nn.tanh(tf.layers.batch_normalization(y, training=training_ind), name='outputs_tanh')
                    y = tf.layers.batch_normalization(y, training=training_ind, name='outputs_linear')
        return tf.squeeze(y[:,:-1,:,:])  
    
#%%    
def leaky_relu(x, leak=0.2, name='leakyrelu'):
            return tf.maximum(x, x * leak, name=name)
        
#%%        
def discriminator(x, training = False):
    training_ind = training > 0.
    y = tf.convert_to_tensor(x)
    y = tf.expand_dims(y, axis=-1)
    with tf.name_scope('disc'):
        for i in range(len(depths_disc)):
            with tf.variable_scope('downconv_{}'.format(i)):
                y = tf.layers.conv2d(y, depths_disc[i], [3, 3], strides=(2, 2), padding='SAME')
                y = leaky_relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_leakyrelu')   
    with tf.variable_scope('classify'):
        batch_size = y.get_shape()[0].value
        reshape = tf.reshape(y, [batch_size, -1, 1])
        y = tf.layers.dense(reshape, 1, activation= None, name='outputs_dense')
#    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
    return y

#%% test disc
#x = tf.random_normal(shape=[1, 256, 16])
#y = discriminator(x)
#sess=tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#y = sess.run(y)
#print(y)