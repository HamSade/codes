#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:58:25 2017
@author: hsadeghi
"""
import tensorflow as tf
import numpy as np
from binary_quantizer import binary_quantizer

#%% General parameters

input_dim = 2**13
down_ratio = 4
bn = True

#coder_depths = [128, 256, 512, 1024]
#disc_depths = [128, 256, 512, 1024]

coder_depths = list(np.array([64, 128, 256, 512])//down_ratio)
disc_depths = list(np.array([64, 128, 256, 512])//down_ratio)

compressed_row = 256 // 16 // down_ratio
compressed_col = 64 // 16 // down_ratio

z_dim = 10  # 4 Kbs


#%%
class Encoder:
    def __init__(self, depths=coder_depths):
        self.depths = [1] + depths
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        if len(outputs.get_shape().as_list()) < 4:
            outputs = tf.expand_dims(outputs, axis=-1)

        with tf.name_scope('coder' + name), tf.variable_scope('coder', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
    
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])  # batch_size * 50 for 2^13
                outputs = tf.layers.dense(reshape, z_dim, name='outputs')
                outputs = tf.tanh(outputs) #to make sure quantizer inputs are in [-1, 1]
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='coder')
        return outputs
    
    
#%%
class Decoder:
    def __init__(self, depths=coder_depths):
        depths.reverse()
        self.depths = depths + [1]
#        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, noise_std, training=False):
        inputs = tf.convert_to_tensor(inputs)        
#        if len(inputs.get_shape().as_list()) < 4:
#            inputs = tf.expand_dims(inputs, axis=-1)
        with tf.variable_scope('coder', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * compressed_row * compressed_col)
                outputs = tf.reshape(outputs, [-1, compressed_row, compressed_col, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                if bn:
                    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = tf.nn.relu(outputs, name='outputs')
                    
            #        noise = tf.random_uniform(tf.shape(inputs), minval=-1.0, maxval=+1.0)
            noise = tf.random_normal(tf.shape(inputs), stddev=noise_std)
            inputs = tf.concat([inputs, noise], axis=-1)        
                    
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME',
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = tf.nn.relu(outputs, name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME',
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = tf.nn.relu(outputs, name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME',
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
                if bn:
                    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = tf.nn.relu(outputs, name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME',
                                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='coder')
        return tf.squeeze(outputs)  #To remove channel size of 1

#%%
class Discriminator:
    def __init__(self, depths=disc_depths):
        self.depths = [1] + depths
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        if len(outputs.get_shape().as_list()) < 4:
            outputs = tf.expand_dims(outputs, axis=-1)
        with tf.name_scope('disc' + name), tf.variable_scope('disc', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                if bn:
                    outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 1, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')
        return outputs

#%%
class Coder:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.enc = Encoder()
        self.dec = Decoder()
        self.disc = Discriminator()
#        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def __call__(self, X, noise_std, training = 1.0):
        
        training_bool = tf.less(training, 0.5)
        
        #        in_q = tf.clip_by_value(enc_output, clip_value_min = -0.999,
#                             clip_value_max = +0.999) 
        in_q = self.enc(X, training = training_bool)
      
        bits = binary_quantizer(in_q, mode=training)
        coder_output = self.dec(bits, noise_std, training=training_bool)
        
        d_real = self.disc(X, training = training_bool)
        d_fake = self.disc(coder_output, training = training_bool)
        
        return coder_output, d_real, d_fake, bits
    