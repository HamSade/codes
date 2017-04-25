#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:27:02 2017

@author: hsadeghi
"""

# Next plans
# Beta and scale used in batch_norm (especially in full_layer) need to be shared among all the steps

#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data
#for stochastic Neurons
#import stochastic_neuron as sn

#import tflearn
#from rnn_layer import rnn_layer


#from tensorflow.python.ops import rnn, rnn_cell



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
training_epochs = 1
batch_size = 128

# Network Parameters

drop_out_p=0.8;


full_width=256;
rnn_width=512;
binary_width=12;
num_steps=2;

display_step = 100

#%% ##############################################################################
# input, weights and biases

X = tf.placeholder("float", [None, input_dim])
#initial_state= tf.placeholder("float", [2, None, rnn_width])
#n_time_samples= batch_size #tf.placeholder("int32", [None])


std_weight=( 2.0 / max( [full_width, rnn_width] ))**0.5;
std_bias=0.01;



#%%
def w_b_gen(shape_param, stddev_param):
    
    weight= tf.Variable(tf.random_normal(shape_param, mean=0.0, stddev=stddev_param)); 
    #bias= tf.Variable(tf.random_normal([out_dim],  mean=0.0, stddev=std_bias));
    
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

def rnn_block(num_neurons):
    
    lstm = tf.contrib.rnn.BasicLSTMCell(num_neurons, state_is_tuple=True)
   
#    lstm = rnn_cell.DropoutWrapper(lstm, inout_keep_prob=drop_out_p, output_keep_prob=drop_out_p)
    
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 2, state_is_tuple=True)
    
    return stacked_lstm
 
    
 
#%%##############################################################################
#%% Full layer

def full_layer (x, w):
    
    layer = batch_norm(x, w)
    layer = tf.nn.tanh(layer);  
    
    return layer
        
#%%##############################################################################
# Building the encoder

rnn_enc = rnn_block(rnn_width)
rnn_dec = rnn_block(rnn_width)


def encoder(x, init_state, rnn_enc):
    
 
    enc_full = full_layer (x, weights['enc_full'])


#    print (enc_full.get_shape()[0].value)  # None
#    print (enc_full.get_shape()[1].value)  # 256
    
#    enc_full = tf.reshape(enc_full, [1,-1, full_width] )
###    
####    print (enc_full.get_shape()[0].value)  # None
####    print (enc_full.get_shape()[1].value)  # 1
####    print (enc_full.get_shape()[2].value)  # 256
###    
#    enc_full = tf.transpose(enc_full, [1,0,2])
#    
#    enc_full = tf.split(enc_full, 1, 0)
##        
#    enc_full = tf.reshape(enc_full, [-1, full_width])
##
#    print (enc_full.get_shape()[0].value)  # None
#    print (enc_full.get_shape()[1].value)  # 256!!!
        
    rnn_output , state = rnn_enc(enc_full,init_state)
#    rnn_output , enc_state = tf.nn.dynamic_rnn( rnn_enc, enc_full, initial_state=init_state)


    print (rnn_output.get_shape()[0].value)  #128 = batch_size
    print (rnn_output.get_shape()[1].value)  #512
     
    
#    rnn_output = tf.transpose(rnn_output, [1, 0, 2])
#    rnn_output = tf.gather(rnn_output, int(rnn_output.get_shape()[0]) - 1)
#    rnn_output=rnn_output[-1]
    
#    rnn_output = tf.reshape(tf.concat(rnn_output, 1), [-1, rnn_width])
    
    # rnn_1=batch_norm(rnn_1, weights['enc_rnn_2'])
    
#    print (rnn_output.get_shape()[0].value)  #128
#    print (rnn_output.get_shape()[1].value)  #512
    
    full_middle= full_layer (rnn_output, weights['middle'])
    
#    print (full_middle.get_shape()[0].value)  #128
#    print (full_middle.get_shape()[1].value)  #12

    
    return full_middle, state

#%%    

def decoder(x, init_state, rnn_dec):
    
    # x=batch_norm(x, weights['dec_rnn_1'])
#    
#    print (x.get_shape()[0].value)  #128
#    print (x.get_shape()[1].value)  #12

#    x=tf.reshape(x, [-1,1,binary_width])
    
#    print (x.get_shape()[0].value)  #128
#    print (x.get_shape()[1].value)  #12
#    print (x.get_shape()[2].value)  #
#
#    x = tf.transpose( x , [1,0,2])
#    
#    print (x.get_shape()[0].value)  #1
#    print (x.get_shape()[1].value)  #128
#    print (x.get_shape()[2].value)  #12
#        
#    x = tf.reshape(x, [-1, binary_width])

    rnn_output, state = rnn_dec(x, init_state)


#    print (rnn_output.get_shape()[0].value)  #
#    print (rnn_output.get_shape()[1].value)  #
#    print (rnn_output.get_shape()[2].value)  #

           
#    rnn_output , state = tf.nn.dynamic_rnn( rnn_dec, x, initial_state=init_state)
    
#    rnn_output=rnn_output[-1]

    # rnn_1=batch_norm(rnn_1, weights['dec_rnn_2'])
    
    dec_full = full_layer (rnn_output, weights['dec_full'])
    
    full_out = full_layer (dec_full, weights['out'])
    
    
    return full_out, state

###############################################################s
# Construct the residual model

residue=X # '- Since at the beginning output is zero


#lstm_temp = tf.contrib.rnn.LSTMCell(rnn_width, state_is_tuple=True)
#stacked_lstm_temp = tf.contrib.rnn.MultiRNNCell( [lstm_temp] * 2, state_is_tuple=True)
#state_temp = stacked_lstm_temp.zero_state(batch_size, tf.float32)

#rnn_temp = rnn_block(rnn_width)
#state_temp = rnn_temp.zero_state(batch_size, tf.float32)

state_enc = rnn_enc.zero_state(batch_size, tf.float32)
state_dec = rnn_dec.zero_state(batch_size, tf.float32)



for _ in range(num_steps):
    
    print('iteration', _)
    
    encoder_output, state_enc = encoder(residue, state_enc, rnn_enc)
    
    decoder_output, state_dec = decoder(encoder_output, state_dec, rnn_dec)
    
    residue = X - decoder_output
    



#%% Cost and optimization setup

y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow( y_true - decoder_output , 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


#%%##############################################################################
# Training
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph, mean=0.0, stddev=1))
#with tf.Session() as sess:
sess=tf.Session();
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

sess.run(init)


# Training cycle
cost_vector=[];

for epoch in range(training_epochs):
    # Loop over all batches
#        small_cost_occurance=0;
    
    for  i in range(n_batch):
        start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
        
#        batch_xs= tf.reshape(batch_xs, [batch_size,input_dim])       
        
#            batch_xs=batch_xs.reshape(n_input, batch_size);
        # Run optimization op (backprop) and cost op (to get loss value)
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

# Plotting results
#plt.plot(cost_vector)

#%%##########################################################################
# Savings network

AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/research codes/April/AE_output.mat",
           AE_output);

sess.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# helper functions

#def unpack_sequence(tensor):
#    """Split the single tensor of a sequence into a list of frames."""
#    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))
#
#def pack_sequence(sequence):
#    """Combine a list of the frames into a single tensor of the sequence."""
#    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])
#
