#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 18:12:27 2017

@author: hsadeghi


I use TFLEARN to build the model

"""



from __future__ import division, print_function, absolute_import

import tensorflow as tf

import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data


#%%
#loading data

mat = si.loadmat('/home/hsadeghi/Downloads/PDA/my_data/com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
# >> median(abs(concat_wav)) = 0.0376

n_data=data.shape[1];
#input_dim =int(fs/25); # Frames size of speech signal
input_dim =100;

n_data=n_data - n_data % input_dim; # make length divisible by input_dim
data=data[0,0:n_data]; # clipping teh rest
# Reshaping data
data=data.reshape([int(n_data/input_dim), input_dim])
n_data=data.shape[0];

#data=tf.cast(data, tf.float32)


training_percentage=90;
n_training= int(np.floor(training_percentage/100.*n_data));
training_data=data[ 0:n_training , : ];

test_data=data[ n_training:n_data, : ];
n_test=test_data.shape[0];



#%% Parameters

learning_rate = 0.001
training_iters = 1000
num_steps = 2 # timesteps
display_step = 100
training_epochs=1;

batch_size = 128
num_batches= 2000; # int(n_training/batch_size)


# Network Parameters
input_dim = 128


full_width=512; # number of neurons in the fully connected par

rnn_width = 512 # number of neurons in the rnn part
num_layers=2

num_binary=16;


#%% tf Graph input
X = tf.placeholder('float', [None, input_dim])

# define states??!! Not sure if required or not!
previous_state_enc=tf.placeholder('float', [rnn_width, num_layers])
previous_state_dec=tf.placeholder('float', [rnn_width, num_layers])





























