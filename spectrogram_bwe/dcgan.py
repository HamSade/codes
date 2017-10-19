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
#depths_disc= [1] + [64, 128, 256, 512]
#depths_reg = [1024, 512, 256, 128]+ [1] 

#%%
coeff = 20
depths_disc= [1] + list( coeff  * np.power(2, np.arange(1,5)) )
depths_reg =   list( coeff * np.power(2, np.arange(4, 0, -1)) ) + [1]

#%%        
def leaky_relu(x, leak=0.2, name='leakyrelu'):
            return tf.maximum(x, x * leak, name=name)

#%%
def regressor_noise(x, training = 0., activation = None):
    y = tf.convert_to_tensor(x)
    training_ind = training > 0.
    # NOise and LB form 2 channels --  no need to expand
#    y = tf.expand_dims(y, axis=-1)  # to indicate #channels = 1
    with tf.variable_scope('reg'):
        for i in range(len(depths_reg)):
            with tf.variable_scope('conv_{}'.format(i)):
                if i < (len(depths_reg) - 1):
                    print('Layer', i)
                    y = tf.layers.conv2d(y, depths_reg[i], [5, 5], strides=(1, 1), padding='SAME', name='reg_outputs_{}'.format(i))
#                    y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
#                    y = leaky_relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
                    y = leaky_relu(y, name='outputs_relu_{}'.format(i))
                    

 
                else:
                    y = tf.layers.conv2d(y, depths_reg[i], [5, 5], strides=(1, 1), padding='SAME', name='reg_outputs_{}'.format(i))
#                    y = tf.nn.tanh(tf.layers.batch_normalization(y, training=training_ind), name='outputs_tanh')
#                    y = tf.layers.batch_normalization(y, training=training_ind, name='outputs_linear')
        if activation != None:
            y = activation(y)
        print('final y shape', y.get_shape().as_list())
        
        return tf.squeeze(y[:,:-1,:,:])  

#%%
def discriminator(x, training = False, activation= tf.sigmoid):
    training_ind = training > 0.
    y = tf.convert_to_tensor(x)
    if len(x.get_shape().as_list()) < 4:
        y = tf.expand_dims(y, axis=-1)  # to add dimension 1 to our inout which is 3 dimensional
    with tf.name_scope('disc'):
        for i in range(len(depths_disc)):
            with tf.variable_scope('downconv_{}'.format(i)):
                y = tf.layers.conv2d(y, depths_disc[i], [5, 5], strides=(2, 2), padding='SAME',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='disc_outputs_{}'.format(i))
                
#                y = tf.nn.relu(y, name='outputs_leakyrelu')
                y = leaky_relu(y, name='outputs_leakyrelu')
#                if i>0:
#                    y = leaky_relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_leakyrelu')   
#                else:
#                    y = leaky_relu(y, name='outputs_leakyrelu')   
    with tf.variable_scope('classify'):
        batch_size = y.get_shape()[0].value
        reshape = tf.reshape(y, [batch_size, -1, 1]) #tf.reshape(outputs, [batch_size, -1])
        y = tf.layers.dense(reshape, 1, activation= activation, name='outputs_dense')
    return y        
#%%
#def regressor(x, training = 0., activation = tf.tanh):
#    training_ind = training > 0.
#    y = tf.convert_to_tensor(x)
##    y = tf.expand_dims(y, axis=-1)  # to indicate #channels = 1
#    with tf.variable_scope('reg'):
#        for i in range(len(depths_reg)):
#            with tf.variable_scope('conv_{}'.format(i)):
#                print('Layer', i)
#                y = tf.layers.conv2d(y, depths_reg[i], [5, 5], strides=(1, 1), padding='SAME')
#                
##                y = leaky_relu( y, name='outputs_relu_{}'.format(i))
#                if i < (len(depths_reg) - 1):  #last layer is linear (Even without BN!!!)
##                    y = tf.nn.relu(y, name='outputs_relu')
##                    y = tf.nn.relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
#                    y = leaky_relu(tf.layers.batch_normalization(y, training=training_ind), name='outputs_relu_{}'.format(i))
#                else:
#                    y = tf.layers.batch_normalization(y, training=training_ind, name='outputs_linear')
#                    y = activation(tf.layers.batch_normalization(y, training=training_ind), name='outputs_tanh')
#        return tf.squeeze(y[:,:-1,:,:])  

#%% test disc
#x = tf.random_normal(shape=[1, 256, 16])
#y = discriminator(x)
#sess=tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#y = sess.run(y)
#print(y)