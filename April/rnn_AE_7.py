#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 17:34:23 2017

@author: hsadeghi
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data
#from tensorflow.python.ops import rnn, rnn_cell

#%%
#loading data
mat = si.loadmat('com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
n_data=data.shape[1];
input_dim =128;
n_data=n_data - n_data % input_dim; # make length divisible by input_dim
data=data[0,0:n_data]; # clipping teh rest
# Reshaping data
data=data.reshape([int(n_data/input_dim), input_dim])
n_data=data.shape[0];
training_percentage=90;
n_training= int(np.floor(training_percentage/100.*n_data));
training_data=data[ 0:n_training , : ];
test_data=data[ n_training:n_data, : ];
n_test=test_data.shape[0];


#%% Parameters
learning_rate   = 0.001
training_epochs = 1
batch_size      = 128
n_batch = int(2 * n_training / batch_size)

full_width   = 128;
rnn_width    = 64;
binary_width = 12;
num_steps    = 16;



display_step = 100

#%% ##############################################################################
# input, weights and biases
X = tf.placeholder("float", [None, input_dim])
#drop_out_p=tf.placeholder("float", [1])


std_weight=( 2.0 / max( [full_width, rnn_width] ))**0.5;
std_bias=0.01;

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


#%%#############################################
###################  RNN layer #################
################################################   
   
def rnn_block(num_neurons):
 
    rnn = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True) 
    rnn = tf.contrib.rnn.MultiRNNCell([rnn] * 2, state_is_tuple=True)
    
    return rnn
    
#%%##############################################################################
#%% Full layer

def full_layer(x, w):
    
    layer = batch_norm(x, w)

#    layer = tf.matmul(x, w)
    layer = tf.nn.tanh(layer);  
    
    return layer

#%%
def batch_norm(x, W):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
    
    scale = tf.Variable(tf.ones([num_neurons]))
    beta = tf.Variable(tf.zeros([num_neurons]))
    
#    scale = tf.ones ([num_neurons])
#    beta  = tf.zeros([num_neurons])
    
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN
        
#%%##############################################################################
# Building the codec

#enc=tf.get_variable()
class codec():
    
    def __init__(self, w):
        
        self.rnn_enc = rnn_block(rnn_width)
        self.rnn_dec = rnn_block(rnn_width)
        
        self.w=w
        
        
    def encoder(self, x, init_state):
         
        enc_full = full_layer (x, self.w['enc_full'])

        try:
            with tf.variable_scope('enc'):
                rnn_output , enc_rnn_state = self.rnn_enc(enc_full,init_state)

        except ValueError:
            with tf.variable_scope('enc', reuse=True):
                tf.get_variable_scope().reuse_variables()
                rnn_output , enc_rnn_state = self.rnn_enc(enc_full,init_state)    
        
        
#        self.enc_rnn_output = tf.reshape(self.enc_rnn_output, [-1, rnn_width])    
    
        full_middle= full_layer (rnn_output, self.w['middle'])
        
        
        return full_middle, enc_rnn_state
    
    ############################################
    
    def decoder(self, x, init_state):
    
#        x = tf.reshape(x, [-1, 1, binary_width])
        
#        with tf.variable_scope('dec', reuse=True):
#        tf.get_variable_scope().reuse_variables()
#        rnn_dec = rnn_block(rnn_width)


        try:
            with tf.variable_scope('dec'):
                rnn_output, dec_rnn_state = self.rnn_dec( x, init_state)

        except ValueError:
            with tf.variable_scope('dec', reuse=True):
                tf.get_variable_scope().reuse_variables()
                rnn_output, dec_rnn_state = self.rnn_dec( x, init_state)
        
        
#        self.dec_rnn_output = tf.reshape(self.dec_rnn_output, [-1, rnn_width])
       
        dec_full = full_layer (rnn_output, self.w['dec_full'])
        
        full_out = full_layer (dec_full, self.w['out'])
            
        return full_out, dec_rnn_state

###############################################################s
# Construct the residual model

residue = X # '- Since at the beginning output is zero


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


#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session()
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

sess.run(init)
# Training cycle
cost_vector=[];

for epoch in range(training_epochs):
   
    for  i in range(n_batch):
        start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
        batch_xs= training_data[ start_ind* int(0.5*batch_size) :\
                                start_ind* int(0.5*batch_size) + batch_size, :];  # 50% overlap 
        
#        batch_xs= tf.reshape(batch_xs, (batch_size,1,input_dim) )       
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        
    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%02d' % (epoch+1),
                  "i:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(c))   
            
            cost_vector+=[c]

print("Optimization Finished!")


#%%##########################################################################
# Testing the network performance
training_error=sess.run(cost, feed_dict={X: training_data})**0.5

y_pred_test, y_true_test, test_error = sess.run([decoder_output, y_true, cost],
                                                feed_dict={X: test_data})

test_error = test_error ** 0.5

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))
#print('architecture ', hid_size)
print('learning_rate= ', learning_rate)
print('num_quantization_steps= ', num_steps)
#%%##########################################################################
# Savings network

AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/April/AE_output.mat",
           AE_output);

sess.close()
