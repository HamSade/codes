#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:16:58 2017

@author: hsadeghi

Added pre_emph and de_emph to rnn_AE_11. Suitsable for testing individual configs

"""

#%%

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as si 

from binary_quantizer import binary_quantizer

from data_loader import pre_emph, de_emph, data_loader

#%%
#loading data
# We use 10 mat files (each 200 MB) for training and one for test 

input_dim =512;
emphasis = True

#%% Parameters

learning_rate = .0001 
training_epochs = 5
batch_size      = 64

full_width   = 1024
rnn_width    = 512
binary_width = 256
num_steps    = 4

dropout_p = 1.0

display_step = 100

print('emphasis', emphasis)

print('input_dim', input_dim)
print('batch_size', batch_size)
print('full_width', full_width)
print('rnn_width', rnn_width)
print('binary_width', binary_width)
print('num_steps', num_steps)
print('dropout_p', dropout_p)


print('learning_rate', learning_rate)
#%% ##############################################################################
# input, weights and biases
X = tf.placeholder("float", [None, input_dim])
#drop_out_p=tf.placeholder("float", [1])
mode = tf.placeholder("float", None)  # for quantizing neurons


std_weight=0.001; #( 2.0 / max( [full_width, rnn_width] ))**0.5;
std_bias=0.0001;

#%%
def w_b_gen(shape_param, stddev_param):
    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
    return weight


#%% Generating weights
weights={}
biases={}

weights['enc_full']  = w_b_gen( [input_dim, full_width]     , std_weight)
weights['middle']    = w_b_gen( [rnn_width, binary_width]   , std_weight)
weights['dec_full']  = w_b_gen( [rnn_width, full_width]     , std_weight)
weights['out']       = w_b_gen( [full_width, input_dim]     , std_weight)

biases['out']        = w_b_gen( [input_dim]     , std_bias)
#%%#############################################
###################  RNN layer #################
################################################   
   
def rnn_block(num_neurons):
 
#    rnn = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True) 
#    rnn = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True)  for _ in range(2)], state_is_tuple=True)
    
    rnn_1 = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True) 
    rnn_2 = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True) 
    rnn= tf.contrib.rnn.MultiRNNCell([rnn_1, rnn_2], state_is_tuple=True)
    
    return rnn
    
#%%##############################################################################
#%% Full layer

def full_layer(x, w, scope):
    
    try:
        with tf.variable_scope(scope, reuse=True):
            tf.get_variable_scope().reuse_variables()
            layer = batch_norm(x, w, scope)
            print('reuse')
    except ValueError:
        with tf.variable_scope(scope):
            print('creation')
            layer = batch_norm(x, w, scope)
#        layer = tf.matmul(x, w)
    layer = tf.nn.tanh(layer);  
    
    return layer

#%%
def batch_norm(x, W, scope):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
          
    scale = tf.get_variable(scope +'scale', shape=[num_neurons], dtype=tf.float32)  #  (tf.ones([num_neurons]))
    beta  = tf.get_variable(scope +'beta', shape=[num_neurons] ,dtype=tf.float32)  

    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN
        
#%%##############################################################################
# Building the codec


b_q=binary_quantizer(tf)


#enc=tf.get_variable()
class codec():
    
    def __init__(self, w):
        
        self.rnn_enc = rnn_block(rnn_width)
        self.rnn_dec = rnn_block(rnn_width)
        
        self.w=w
        self.biases = biases
        
        
    def encoder(self, x, init_state):
         
        enc_full = full_layer (x, self.w['enc_full'], 'enc_full')
        enc_full = tf.nn.dropout(enc_full, dropout_p)

        try:
            
            with tf.variable_scope('enc', reuse=True):
                tf.get_variable_scope().reuse_variables()
                rnn_output , enc_rnn_state = self.rnn_enc(enc_full,init_state)
                
        except ValueError:
            
            with tf.variable_scope('enc'):
                rnn_output , enc_rnn_state = self.rnn_enc(enc_full,init_state)
                
#        self.enc_rnn_output = tf.reshape(self.enc_rnn_output, [-1, rnn_width])    
    
        full_middle= full_layer (rnn_output, self.w['middle'], 'full_middle')
        
        full_middle = b_q(full_middle, mode)
        
        
        return full_middle, enc_rnn_state
    
    ############################################
    
    def decoder(self, x, init_state):
    
#        x = tf.reshape(x, [-1, 1, binary_width])
        
#        with tf.variable_scope('dec', reuse=True):
#        tf.get_variable_scope().reuse_variables()
#        rnn_dec = rnn_block(rnn_width)


        try:
            with tf.variable_scope('dec', reuse=True):
                tf.get_variable_scope().reuse_variables()
                rnn_output, dec_rnn_state = self.rnn_dec( x, init_state)
        except ValueError:
            with tf.variable_scope('dec'):
                rnn_output, dec_rnn_state = self.rnn_dec( x, init_state)
            
        
        
#        self.dec_rnn_output = tf.reshape(self.dec_rnn_output, [-1, rnn_width])
       
        dec_full = full_layer (rnn_output, self.w['dec_full'], 'dec_full')
        dec_full = tf.nn.dropout(dec_full, dropout_p)
        
        
        full_out = full_layer (dec_full, self.w['out'], 'full_out')
#        full_out = tf.add( self.biases['out'], tf.matmul( dec_full,  self.w['out']) )
#        full_out = tf.nn.tanh(full_out)
        
            
        return full_out, dec_rnn_state

###############################################################s
# Construct the residual model

if emphasis:
    X= pre_emph(X)
    
residue =X # '- Since at the beginning AEoutput is zero


AE = codec(weights)

init_enc = AE.rnn_enc.zero_state(tf.shape(X)[0], tf.float32)
init_dec = AE.rnn_dec.zero_state(tf.shape(X)[0], tf.float32)
 
   
#%% The entire AUTOENCODER
#
with tf.variable_scope("ae"):
    
#    tf.get_variable_scope().reuse_variables()
    encoder_output, state_enc = AE.encoder(residue, init_enc)
    decoder_output, state_dec = AE.decoder(encoder_output, init_dec)
    residue = X - decoder_output 
    print('iteration', '1') 


for _ in range(num_steps-1): 
        
    with tf.variable_scope("ae", reuse=True):   
        
        tf.get_variable_scope().reuse_variables()
    
        encoder_output, state_enc = AE.encoder(residue, init_enc)
            
        decoder_output, state_dec = AE.decoder(encoder_output, init_dec)
               
        residue = X - decoder_output
    
        print('iteration', _+2)
            
            
            
        #%% Cost and optimization setup
y_true = X;
cost = tf.reduce_mean( tf.pow( y_true - decoder_output , 2))
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)
#optimizer = tf.train.GradientDescent1024Optimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

sess.run(init)

#file_writer = tf.summary.FileWriter('logs/', sess.graph)

# Training cycle
cost_vector=[];


for epoch in range(training_epochs):
    
#    learning_rate_new = learning_rate_new /2.
    
    for file_idx in range(10):
        
#        file_name = 'com_concat_signal_{}.mat'.format(file_idx)
        file_name = 'clean_{}.mat'.format(file_idx)
        
        training_data, n_training = data_loader(file_name, input_dim, overlap=False)
        
        n_batch = int(n_training / batch_size)
           
        for  i in range(n_batch):
            
            start_ind = np.random.randint(0, n_training-batch_size );  # For shuffling
            
#            batch_xs= training_data[ start_ind* batch_size : (start_ind + 1)* batch_size, :]
            batch_xs= training_data[ start_ind : start_ind + batch_size, :]
            
    #        batch_xs= tf.reshape(batch_xs, (batch_size,1,input_dim) )       
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, mode:0.0})
            
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch", '%02d' % (epoch+1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i+1),
                      "cost=", "{:.9f}".format(c))   
                
                cost_vector+=[c]

print("Optimization Finished!")


#%%##########################################################################
# Testing the network performance

training_data, n_training = data_loader('clean_9.mat', input_dim, overlap=False)
training_error=sess.run(cost, feed_dict={X: training_data, mode:1.0})**0.5

test_data, n_test = data_loader('clean_10.mat', input_dim, overlap=False)

y_pred_test, y_true_test, test_error = sess.run([decoder_output, y_true, cost],
                                                feed_dict={X: test_data, mode:1.0})

test_error = test_error ** 0.5


print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#print('learning_rate= ', learning_rate)
#print('num_quantization_steps= ', num_steps)
#%%##########################################################################
# Savings network

AE_output={};


#AE_output['y_pred_test']=y_pred_test
#AE_output['y_true_test']=y_true_test
    
if emphasis:
    y_pred_test = de_emph(y_pred_test, input_dim)
    y_true_test = de_emph(y_true_test, input_dim)

AE_output['y_pred_test']=y_pred_test
AE_output['y_true_test']=y_true_test
    

si.savemat("/home/hsadeghi/Dropbox/May/past_codes/rnn_AE_output.mat",
           AE_output);

sess.close()
