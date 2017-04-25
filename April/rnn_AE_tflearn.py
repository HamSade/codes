#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:29:39 2017

@author: hsadeghi
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data
#for stochastic Neurons
#import stochastic_neuron as sn

import tflearn

#%%
#loading data
mat = si.loadmat('com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
# >> median(abs(concat_wav)) = 0.0376

n_data=data.shape[1];
#input_dim =int(fs/25); # Frames size of speech signal
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
drop_out_p=0.5;

training_epochs = 1
batch_size = 128

# Network Parameters

#hid_size=[512,512];

full_width=512
rnn_width=512;
binary_width=16;
num_steps=2;
dropout_p=0.5;

display_step = 100




#%% layers definition

def batch_normalization(x):
    
    x_bn= tflearn.layers.normalization.batch_normalization (x, beta=0.0,epsilon=1e-05);
   
    return x_bn

#%% Full layer

def full_layer (x, width):
    
    output= tflearn.fully_connected(x, width, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
    
    output = tflearn.dropout(output, dropout_p)
    
    
#%% RNN block

def rnn_block(x, n_neurons, init_state):
       
    #    init_state = lstm.zero_state(batch_size, tf.float32)
        
#    x=batch_normalization(x);

    lstm = tf.contrib.rnn_cell.LSTMCell(n_neurons, state_is_tuple=False)
    lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, input_keep_prob=dropout_p,
                                         output_keep_prob=dropout_p)
        
    
    lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=False)  #number of layers is 2    

    output, state = tf.contrib.rnn.static_rnn(lstm, x, initial_state=init_state)    
    
        
    return output, state
    


#%% TFLEARN Network buildup

#X = tf.placeholder("float", [None, input_dim])
X = tflearn.input_data(shape=[batch_size, num_steps, input_dim])


def encoder(x, init_state):
    
#    x=tf.reshape(x, shape=[-1, input_dim])
    
    full_in = full_layer(x, full_width)

    full_in = tf.reshape(full_in, shape=[batch_size, 1, full_width])
    
    rnn_output, state_enc= rnn_block(full_in, rnn_width , init_state)

    rnn_output= tf.reshape(rnn_output, shape=[-1, rnn_width])
    
    full_middle= full_layer(rnn_output, binary_width)

    return full_middle, state_enc
 
#%%

def decoder(x, init_state):
    
    x= tf.reshape(x, shape=[batch_size, 1, binary_width])
    
    rnn_output, state_dec= rnn_block(x, rnn_width , init_state)
    
    rnn_output= tf.reshape(rnn_output, shape=[-1, rnn_width])
    
    full_last = full_layer(rnn_output, full_width)

    full_out = tf.reshape(full_last, shape=[batch_size, 1, full_width])

    return full_out, state_dec


#%%
## Construct the residual model


input_tensor=X;


# zeros initial state generation
temp = tf.contrib.rnn_cell.LSTMCell(rnn_width, state_is_tuple=False)
lstm = tf.nn.rnn_cell.DropoutWrapper(temp, input_keep_prob=dropout_p,
                                     output_keep_prob=dropout_p)
temp = tf.nn.rnn_cell.MultiRNNCell([temp] * 2, state_is_tuple=False) 
 
state_enc= state_dec = temp.zero_state(batch_size, tf.float32)


for _ in range(num_steps):
    
    encoder_output, state_enc = encoder(input_tensor, state_enc)
    
    decoder_output, state_dec = decoder(encoder_output, state_dec)
    
    input_tensor= input_tensor - decoder_output
    
    

## Ground truth
y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!

## Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow(y_true - y_pred[str(num_quantization_steps)], 2))
#
##optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
##optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)
#
#
##%%##############################################################################
## Initializing the variables
#init = tf.global_variables_initializer()
#
## Launch the graph, mean=0.0, stddev=1))
#
##with tf.Session() as sess:
##sess=tf.Session();
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#
#sess.run(init)
#
#
## Training cycle
#cost_vector=[];
#
#for epoch in range(training_epochs):
#    # Loop over all batches
##        small_cost_occurance=0;
#    
#    for  i in range(n_batch):
#        start_ind=np.random.randint(0, n_training-batch_size );  # For shuffling
#        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
##            batch_xs=batch_xs.reshape(n_input, batch_size);
#        # Run optimization op (backprop) and cost op (to get loss value)
#        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
#    # Display logs per epoch step
#        if i % display_step == 0:
#            print("Epoch:", '%02d' % (epoch+1),
#                  "i:", '%04d' % (i+1),
#                  "cost=", "{:.9f}".format(c))   
#            
#            cost_vector+=[c]
#
#print("Optimization Finished!")
#
##%%##########################################################################
## Testing the network performance
#
#training_error=sess.run(cost, feed_dict={X: training_data})**0.5
#
#y_pred_test, y_true_test, test_error = sess.run([y_pred[str(num_quantization_steps)], y_true, cost], feed_dict={X: test_data})
#
#test_error=test_error**0.5
#
##_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
#print( 'training_error', "{:.9f}".format(training_error))
#print( 'test_error', "{:.9f}".format(test_error))
#
#
#print('architecture ', hid_size)
#print('learning_rate= ', learning_rate)
#print('n_hidden_binary= ', n_hidden_binary)
#print('num_quantization_steps= ', num_quantization_steps)
#
## Plotting results
##plt.plot(cost_vector)
#
##%%##########################################################################
## Savings network
##saver = tf.train.Saver()
##save_path = saver.save(sess, "/home/hsadeghi/Dropbox/research codes/",
##                       "binary_full_2step_4bit_AE.ckpt")
#
##    print("Model saved in file: %s" % save_path)
#AE_output={};
#AE_output['y_pred_test']=y_pred_test;
#AE_output['y_true_test']=y_true_test;
#
#si.savemat("/home/hsadeghi/Dropbox/research codes/April/AE_output.mat",
#           AE_output);
#
#           
#           
#sess.close()
#
#
##%% Building graph
#
##writer = tf.summary.FileWriter('/Dropbox/research codes/log', sess.graph)




