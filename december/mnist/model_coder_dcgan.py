#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:31:06 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
#from binary_quantizer import binary_quantizer

#%% General parameters
input_dim = 2**13
std_dev_0 = 0.001
bn = False

#coder_depths = [128, 256, 512, 1024]
#disc_depths = [128, 256, 512, 1024]

#encoder_depths =  list(np.array([128, 256, 512, 1024]))
#disc_depths = list(np.array([64, 128, 256, 512]))
encoder_depths =  list(np.array([128, 256]))
disc_depths = list(np.array([64, 128]))

compressed_row = 32 // 4
compressed_col = 32 // 4
z_dim = 50

#%%
class Discriminator:
    def __init__(self, depths=disc_depths):
        self.depths = [1] + depths
        print('Discriminator depths', self.depths)
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        outputs = tf.convert_to_tensor(inputs)
        print('disc output size at input', outputs.get_shape().as_list()) 
        if len(outputs.get_shape().as_list()) < 4:
            outputs = tf.expand_dims(outputs, axis=-1)
            print('disc output size at input', outputs.get_shape().as_list())            
        with tf.name_scope('disc' + name), tf.variable_scope('disc', reuse=self.reuse):
            for i in range(1,len(self.depths)):
                with tf.variable_scope('disc_conv_'+str(i)):
       
                    ### DCGAN METHOD
                    outputs = tf.layers.conv2d(outputs, self.depths[i], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev_0))
                    if bn:
                        outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                    else:
                        outputs = leaky_relu(outputs, name='outputs')
                    
                    print('disc output size at layer' +  str(i), outputs.get_shape().as_list()) 
            with tf.variable_scope('disc_classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1])
                outputs = tf.layers.dense(reshape, 1, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='disc')
        return outputs

   
#%%
class Generator:
    def __init__(self, depths=encoder_depths):
        depths.reverse()
        self.depths = depths + [1]
        print('Decoder depths', self.depths)
        self.reuse = False
    def __call__(self, inputs, training=False):
        inputs = tf.convert_to_tensor(inputs)
        print('Generator input size ', inputs.get_shape().as_list())         
        with tf.variable_scope('coder', reuse=self.reuse):
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * compressed_row * compressed_col)
                outputs = tf.reshape(outputs, [-1, compressed_row, compressed_col, self.depths[0]])
                if bn:
                    outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                else:
                    outputs = tf.nn.relu(outputs, name='outputs')   
                print('Generator size at input ', outputs.get_shape().as_list()) 
            # deconvolution (transpose of convolution) x 4
            for i in range(1,len(self.depths)):
                with tf.variable_scope('deconv'+str(i)):
                    ### DCGAN methods
                    if i<len(self.depths)-1:
                        outputs = tf.layers.conv2d_transpose(outputs, self.depths[i], [5, 5], strides=(2, 2), padding='SAME')
                        outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
                    else: # i==len(self.depths)-1:
                        outputs = tf.layers.conv2d_transpose(outputs, self.depths[i], [5, 5], strides=(2, 2), padding='SAME',
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=std_dev_0), name='linear_layer')         
                    
                    print('Generator size at layer ' + str(i) , outputs.get_shape().as_list()) 
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='coder')
        return tf.squeeze(outputs)  #To remove channel size of 1
    
#%%
class GAN:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.gen = Generator()
        self.disc = Discriminator()
        
    def __call__(self, X, training = 1.0):
        
        training_bool = tf.less(training, 0.5)  
        
        noise = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
#        noise = tf.random_normal( [self.batch_size, self.z_dim], stddev= 1.0)
        gen_output = self.gen(noise, training = training_bool)
        
        print('coder_output', gen_output.get_shape().as_list())
        print('X', X.get_shape().as_list())
        
        d_real = self.disc(X, training = training_bool)
        d_fake = self.disc(gen_output, training = training_bool)
        
        print('d_real shape', d_real.get_shape().as_list())
        print('d_fake shape', d_fake.get_shape().as_list())
        
        
        return gen_output, d_real, d_fake   
 