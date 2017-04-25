#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:54:49 2017

@author: hsadeghi


Implemented LSTM manually!!! So slow....


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


#from rnn_layer import rnn_layer

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


#%%
def w_b_gen(shape_param, stddev_param):
    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
    return weight

#%% Initialization parameters

std_weight=(2.0/max([rnn_width, full_width]))**0.5;
std_bias=0.01

weights={}
biases={}

weights['enc_full']  = w_b_gen( [input_dim, full_width]     , std_weight)
weights['middle']    = w_b_gen( [rnn_width, binary_width]   , std_weight)

weights['enc_rnn_1']  = w_b_gen( [4, full_width+rnn_width]     , std_weight)
weights['enc_rnn_2']  = w_b_gen( [4, rnn_width+rnn_width]      , std_weight)
weights['dec_rnn_1']  = w_b_gen( [4, binary_width+rnn_width]   , std_weight)
weights['dec_rnn_2']  = w_b_gen( [4, rnn_width+rnn_width]      , std_weight)


biases['enc_rnn_1']  = w_b_gen( [4, rnn_width]     , std_weight)
biases['enc_rnn_2']  = w_b_gen( [4, rnn_width]     , std_weight)
biases['dec_rnn_1']  = w_b_gen( [4, rnn_width]     , std_weight)
biases['dec_rnn_2']  = w_b_gen( [4, rnn_width]     , std_weight)

weights['dec_full']  = w_b_gen( [rnn_width, full_width]     , std_weight)
weights['out']       = w_b_gen( [full_width, input_dim]     , std_weight)

#%% layers definition

def batch_normalization(x):
    
    x_bn= tflearn.layers.normalization.batch_normalization (x, beta=0.0,epsilon=1e-5);
   
    return x_bn

#%% Full layer

def full_layer (x, width):
    
    output= tflearn.fully_connected(x, width, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
    
    output = tflearn.dropout(output, dropout_p)
    
    return output
    
    
#%% RNN block


def rnn_layer(x, previous_state, w, b):
     
     # x is [num_batches , input_dim]
     # weights is [ 4, in_dim + num_neurons]
     # biases is [4, num_neorons]
     # previous_state is [2 , num_neurons] includes both h & c 
     
     h= previous_state[0] # All vectors are row vectors
     c= previous_state[1]

     
     output=[]
     
     for idx in range(x.get_shape()[0].value):
         
         # xx=tf.transpose(x)
         h_temp = tf.transpose( tf.reshape( h, [1, rnn_width]) );
         x_temp = tf.transpose( tf.reshape( x[idx], [1, x.get_shape()[1].value ]));
         h_x = tf.concat([h_temp, x_temp],0);
         
         Th = tf.matmul( w, h_x ) 
         
         i = tf.sigmoid( tf.add( Th[0,:] , b[0,:]) )
         f = tf.sigmoid( tf.add( Th[1,:] , b[1,:]) )
         o = tf.sigmoid( tf.add( Th[2,:] , b[2,:]) )
         g = tf.tanh(    tf.add( Th[3,:] , b[3,:]) )

         c= tf.add ( tf.multiply(f, c) , tf.multiply(i, g) )    
         h= tf.multiply ( o, tf.tanh(c) )    
         
         state = tf.stack([h,c])
         
         h_temp = tf.reshape( h, [1, rnn_width]);
         output += [h]
         
     output=tf.stack(output)
         
         
#     print(output.get_shape()[0].value)
#     print(output.get_shape()[1].value)
         
     return output, state


#%% TFLEARN Network buildup

X = tf.placeholder("float", shape=[batch_size, input_dim])
#X = tflearn.input_data(shape=[None, 1,  input_dim])


def encoder(x, init_state_1, init_state_2, weights):
    
#    x=tf.reshape(x, shape=[-1, input_dim])
        
    full_in = full_layer(x, full_width)

#    full_in = tf.reshape(full_in, shape=[-1, 1, full_width])
    
    rnn_1, state_enc_1 = rnn_layer(full_in, init_state_1, weights['enc_rnn_1'], biases['enc_rnn_1'])
    
    rnn_2, state_enc_2 = rnn_layer(rnn_1 , init_state_2, weights['enc_rnn_2'], biases['enc_rnn_2'])

#    rnn_2= tf.reshape(rnn_2, shape=[-1, rnn_width])
    
    full_middle= full_layer(rnn_2, binary_width)

    return full_middle, state_enc_1, state_enc_2
 
#%%

def decoder(x, init_state_1, init_state_2, weights):
    
#    x= tf.reshape(x, shape=[-1, 1, binary_width])

    rnn_1, state_dec_1= rnn_layer(x, init_state_1, weights['dec_rnn_1'], biases['dec_rnn_1'])    

    rnn_2, state_dec_2= rnn_layer(rnn_1, init_state_2, weights['dec_rnn_2'], biases['dec_rnn_2'])
    
#    rnn_2= tf.reshape(rnn_2, shape=[-1, rnn_width])
    
    full_last = full_layer(rnn_2, full_width)
    
    full_out = full_layer(full_last, input_dim)

#    full_out = tf.reshape(full_last, shape=[batch_size, 1, input_dim])

    return full_out, state_dec_1, state_dec_2


#%%
## Construct the residual model


# zeros initial state generation
state_enc_1 = tf.zeros([2, rnn_width])
state_enc_2 = state_dec_1 = state_dec_2 = state_enc_1

residue=X # '- zero output' in fact


for _ in range(num_steps):
    
    print('iteration', _)
    
    encoder_output, state_enc_1, state_enc_2 = encoder(residue, state_enc_1, state_enc_2, weights)
    
    decoder_output, state_dec_1, state_dec_2 = decoder(encoder_output, state_dec_1, state_dec_2, weights)
    
    residue= X - decoder_output
    
    

## Ground truth
y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!

## Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow(y_true - decoder_output, 2))


##optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
##optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


##%%##############################################################################
## Initializing the variables
init = tf.global_variables_initializer()
#
## Launch the graph, mean=0.0, stddev=1))
#
##with tf.Session() as sess:
##sess=tf.Session();
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#
sess.run(init)
#
#
## Training cycle
cost_vector=[];

for epoch in range(training_epochs):
    # Loop over all batches
#        small_cost_occurance=0;
    
    for  i in range(n_batch):
        
        start_ind=np.random.randint(0, n_training-batch_size );  # For shuffling
        
        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
#        batch_xs=batch_xs.reshape([batch_size, 1, input_dim]);
        
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%02d' % (epoch+1),
                  "i:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(c))   
            
            cost_vector+=[c]

print("Optimization Finished!")
#
##%%##########################################################################
## Testing the network performance

training_error=sess.run(cost, feed_dict={X: training_data})**0.5  #RootMeanSquare value

y_pred_test, y_true_test, test_error = sess.run([decoder_output, y_true, cost], feed_dict={X: test_data})

test_error=test_error**0.5

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


print('architecture ', [full_width, rnn_width, rnn_width, binary_width])
print('learning_rate= ', learning_rate)
print('num_quantization_steps= ', num_steps)
#
## Plotting results
##plt.plot(cost_vector)
#
##%%##########################################################################
## Savings network
#saver = tf.train.Saver()
#save_path = saver.save(sess, "/home/hsadeghi/Dropbox/research codes/",
#                       "binary_full_2step_4bit_AE.ckpt")

#    print("Model saved in file: %s" % save_path)
AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/research codes/April/AE_output.mat",
           AE_output);
#
#           
#           
sess.close()
