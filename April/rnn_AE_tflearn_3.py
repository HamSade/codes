#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:29:42 2017

@author: hsadeghi

This version does not do weight sharing


"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data
#from tensorflow.python.ops import rnn, rnn_cell
import tflearn

#%%
#loading data
mat = si.loadmat('com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
n_data=data.shape[1];
input_dim =512;
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
learning_rate = 0.001
training_epochs = 1
batch_size = 50
n_batch = int(n_training/batch_size)-1

#dropout_p=(0.8, 1);

full_width=512;
rnn_width=512;
binary_width = 32;
num_steps=8;
display_step = 100


# To check if updated through dropbox
print('Frame size= ', input_dim)
print('architecture= ', [full_width, rnn_width, rnn_width, binary_width] )
print('Number of steps= ', num_steps)

#%% ##############################################################################
# input, weights and biases
X = tf.placeholder("float", [None, input_dim])

#drop_out_p=tf.placeholder("float", [1])


#std_weight=( 2.0 / max( [full_width, rnn_width] ))**0.5;
#std_bias=0.01;
#
##%%
#def w_b_gen(shape_param, stddev_param):
#    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
#    return weight
#
#
##%%
def batch_norm(x):
    
    x_bn= tflearn.layers.normalization.batch_normalization (x,epsilon=1e-05);
   
    return x_bn



#%% Full layer

def full_layer (x, width):
    
    output = tflearn.fully_connected(x, width)
    
    output = batch_norm(output)
#    output = tflearn.dropout(output, dropout_p) 
    
    return output


#%%

def rnn_layer(x, init_state, width):
    
    output, state = tflearn.layers.recurrent.lstm (x, width,
                                               #dropout=dropout_p,
                                               weights_init='truncated_normal',
                                               forget_bias=1.0, return_seq=True,
                                               return_state=True, 
                                               initial_state=init_state)
    return output, state 
    
    
    
#%%

#with tf.variable_scope('enc'):

def encoder(x, init_state):
     
    enc_full = full_layer (x, full_width)
    
    enc_full = tf.reshape(enc_full, [-1,1,full_width])
    

    rnn_output, rnn_state_1 = rnn_layer(enc_full, init_state[0], rnn_width)
 
    rnn_output, rnn_state_2 = rnn_layer(rnn_output, init_state[1], rnn_width)


    rnn_output = tf.reshape(rnn_output, [-1, rnn_width])
    
#    print (rnn_output.get_shape()[0].value)  #64 = batch_size
#    print (rnn_output.get_shape()[1].value)  #512

    full_middle= full_layer (rnn_output, binary_width)
    
    rnn_state = [rnn_state_1, rnn_state_2]
    
    return full_middle, rnn_state

#%%    

#with tf.variable_scope('dec'):

    
def decoder(x, init_state):

    x = tf.reshape(x, [-1,1,binary_width])
    
    rnn_output, rnn_state_1 = rnn_layer(x, init_state[0], rnn_width)
    
    rnn_output, rnn_state_2 = rnn_layer(rnn_output, init_state[1], rnn_width)
         
#    rnn_output = tf.transpose(rnn_output, [1,0, rnn_width])

    rnn_output = tf.reshape(rnn_output, [-1, rnn_width])                                                  
   
    dec_full = full_layer (rnn_output, full_width)
    
    full_out = full_layer (dec_full, input_dim)
    
    rnn_state = [rnn_state_1, rnn_state_2]
        
    return full_out, rnn_state

###############################################################s
# Construct the residual model

residue = X # '- Since at the beginning output is zero


init_enc = [( tf.zeros([tf.shape(X)[0],rnn_width], tf.float32) ,
             tf.zeros([tf.shape(X)[0],rnn_width], tf.float32) )]*2
    
init_dec = init_enc

    
for _ in range(num_steps):
    
    #    print('iteration', _)    
    #    with tf.variable_scope("foo") as foo:        
    #        foo.reuse_variables()    
    
#    with tf.variable_scope('enc') as enc:
#        enc.reuse_variables()
    encoder_output, state_enc = encoder(residue, init_enc)
    
#    with tf.variable_scope('dec') as dec:
#        dec.reuse_variables()
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
print('architecture ', [full_width, rnn_width, rnn_width, binary_width] )
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
