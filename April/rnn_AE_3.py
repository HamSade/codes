#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:25:35 2017

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
n_batch = 5000;  #int(n_training/batch_size)
learning_rate = 0.01
training_epochs = 1
batch_size = 64

full_width=101;
rnn_width=512;
binary_width=12;
num_steps=2;
display_step = 100

#%% ##############################################################################
# input, weights and biases
X = tf.placeholder("float", [batch_size, input_dim])
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


#%%
# Batch normalization
def batch_norm(x, W):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)   # x * W
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
    scale = tf.Variable(tf.ones([num_neurons]))
    beta = tf.Variable(tf.zeros([num_neurons]))
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN

#%% RNN layer%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class rnn_block:
    def __init__(self, num_neurons):
        
        rnn = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True) 
        rnn = tf.contrib.rnn.MultiRNNCell([rnn] * 2, state_is_tuple=True)
        
        return rnn

#%%##############################################################################
#%% Full layer

def full_layer (x, w):
    
    layer = batch_norm(x, w)
    layer = tf.nn.tanh(layer);  
    
    return layer
        
#%%##############################################################################
# Building the encoder


#with tf.variable_scope('enc'):
rnn_enc = rnn_block(rnn_width)   
#with tf.variable_scope('dec'):
rnn_dec = rnn_block(rnn_width)
    

#%%

#enc=tf.get_variable()

def encoder(x, init_state):
     
    enc_full = full_layer (x, weights['enc_full'])

#    print (enc_full.get_shape()[0].value)  # 64
#    print (enc_full.get_shape()[1].value)  # 101
    
#    enc_full = tf.reshape(enc_full, [-1,1, full_width] )

#    enc_full = tf.transpose(enc_full, [1,0,2])
#    
#    enc_full = tf.split(enc_full, 1, 0)
##        
#    enc_full = tf.reshape(enc_full, [-1, full_width])
    
#    with tf.variable_scope('enc') as enc_var:
   
    rnn_output , rnn_state = rnn_enc(enc_full, init_state)
#    rnn_output , enc_state = tf.nn.dynamic_rnn( rnn_enc, enc_full, initial_state=init_state)

#    print (rnn_output.get_shape()[0].value)  #64 = batch_size
#    print (rnn_output.get_shape()[1].value)  #512

    full_middle= full_layer (rnn_output, weights['middle'])
    
    return full_middle, rnn_state

#%%    

def decoder(x, init_state):

#    rnn_dec=rnn_block(rnn_width)
    
    rnn_output, rnn_state = rnn_dec(x, init_state)
   
    dec_full = full_layer (rnn_output, weights['dec_full'])
    
    full_out = full_layer (dec_full, weights['out'])
        
    return full_out, rnn_state

###############################################################s
# Construct the residual model

residue=X # '- Since at the beginning output is zero

#lstm_temp = tf.contrib.rnn.LSTMCell(rnn_width, state_is_tuple=True)
#stacked_lstm_temp = tf.contrib.rnn.MultiRNNCell( [lstm_temp] * 2, state_is_tuple=True)
#initial_state = rnn_enc.zero_state(batch_size, tf.float32)

init_enc = rnn_enc.zero_state(batch_size, tf.float32)
init_dec = rnn_dec.zero_state(batch_size, tf.float32)


  
for _ in range(num_steps):
    
    #    print('iteration', _)    
    #    with tf.variable_scope("foo") as foo:        
    #        foo.reuse_variables()    
    
    encoder_output, state_enc = encoder(residue, init_enc)
    
    decoder_output, state_dec = decoder(encoder_output, init_dec)
    
    residue = X - decoder_output

#%% Cost and optimization setup

y_true = X;
cost = tf.reduce_mean( tf.pow( y_true - decoder_output , 2))
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session();
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

test_error=test_error**0.5

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

si.savemat("/home/hsadeghi/Dropbox/research codes/April/AE_output.mat",
           AE_output);

sess.close()
