#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 14:39:40 2017

@author: hsadeghi

Deep highway architecture
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as sio 
from data_loader import data_loader, data_parser #, data_parser_2
#from data_loader import apply_imdct, band_merge #pre_emph, de_emph, 
#from conv_GAN import auto_encoder
from highway_ops import deep_highway, deep_highway_with_carrier
#import matplotlib.pyplot as plt
#matplotlib inline

#%% Main parameters

input_dim = 2 ** 10 
overlap= int(input_dim/2**4)

gpu_frac = 1.0

apply_mask = True # Whether to input zero-padded test input

layer_depths = [input_dim + 2 * overlap] * 3 # [ 2* (input_dim + 2 * overlap) ] + [input_dim + 2 * overlap] * 2
cost_lambda = 1.

n_batch = 1000
training_epochs = 32
num_training_files = 10

#n_batch = 1
#training_epochs = 1
#num_training_files = 1

learning_rate_init = 0.01
lr_dec_fac = 1.1
noise_std_init = [0.001]
noise_dec_fac = np.sqrt(1.0)

#%% Other Parameters
batch_size = 64
display_step = 100

dropout_p = 1.

print('layer_depths', layer_depths)
print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('cost_lambda', cost_lambda)
print('overlap', overlap)
print('input_dim', input_dim)
print('batch_size', batch_size)

#%% ##############################################################################
X = tf.placeholder("float", [3, batch_size, input_dim + overlap * 2])
#drop_out_p=tf.placeholder("float", [1])
#mode = tf.placeholder("float", None)
learning_rate = tf.placeholder("float", None) 
noise_std = tf.placeholder("float", 1)      

#%%##############################################################################
# Building the model
X_l_recon = X[0]
X_h = X[1]
X_l = X[2]
x_real = X_l + X_h

#%% 
noise = tf.random_normal(shape = X_l_recon.get_shape().as_list(), stddev = noise_std)
#x_in_highway = tf.add(X_l_recon, noise)
#x_in_highway = tf.concat([X_l_recon, noise], axis = -1)  # 128 x 2304
x_in_highway = (X_l_recon, noise)
y_ae_h = deep_highway_with_carrier(x_in_highway, layer_depths)
x_reconstructed = X_l + y_ae_h 

#%% Cost & Optimizers definitions
cost_high = tf.reduce_mean(tf.square(tf.subtract(X_h , y_ae_h)))
#cost_high = tf.reduce_mean(tf.abs(tf.subtract(X_h , y_ae_h)))
#cost_low = tf.reduce_mean(tf.square(tf.subtract(X_l , y_ae_h)))
#cost =  cost_high + cost_lambda * cost_low
cost = cost_high
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#opt = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#%% Variables
t_vars = tf.trainable_variables()
for x in t_vars:
    print('all_variables', x.name)
    
#%%##############################################################################
# Training
init = tf.global_variables_initializer()
#sess=tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, gpu_options=gpu_options))
sess.run(init)

#%% Training cycle

cost_list = []
for epoch in range(training_epochs):
    training_data = [[]] * num_training_files
    reading_phase = True
    for file_idx in range(num_training_files): 
        if reading_phase:
            training_data [file_idx] = data_loader(file_idx, input_dim)
        for  i in range(n_batch):
            #pre_emph is zero since we do it on X if we want separatrely            
            batch_xs = data_parser(training_data[file_idx], input_dim, batch_size,
                                   overlap = overlap)         
            
            _, cost_ = sess.run([opt, cost], feed_dict={X: batch_xs,
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init})
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch:", '%02d' % (epoch + 1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i + 1),
                      "cost =", "{:.9f}".format( (10**4) * cost_))      
        learning_rate_init = max(0.0001, learning_rate_init /lr_dec_fac)
        noise_std_init[0] =    noise_std_init[0]/ noise_dec_fac  
    reading_phase = False
print("Optimization Finished!")

#%%##########################################################################
# Training error calculation
training_data= training_data[9] #data_loader(9, input_dim)
training_error = 0
avg_num = 50
for i in range(avg_num):
    sampled_x = data_parser(training_data, input_dim, batch_size, overlap=overlap)
    training_error += sess.run(cost, feed_dict={X: sampled_x, noise_std : [0.]})
training_error = (training_error/ avg_num) ** 0.5

#%% Test error calculation
test_data = data_loader(10, input_dim)    
test_error = 0
avg_num = 50
y_pred_test = np.zeros([avg_num*(batch_size), input_dim + overlap *2])
y_true_test = np.zeros([avg_num*(batch_size), input_dim + overlap *2]) 
y_h_true = np.zeros([avg_num*(batch_size), input_dim + overlap *2])
y_h_pred = np.zeros([avg_num*(batch_size), input_dim + overlap *2]) 
for i in range(avg_num):
    sampled_x = data_parser(test_data, input_dim, batch_size, overlap=overlap, apply_mask=apply_mask)  
    X_h_ , y_ae_h_,\
    y_true_test_, y_pred_test_,\
    test_error_ = sess.run([X_h , y_ae_h, x_real, x_reconstructed, cost], feed_dict={X: sampled_x, noise_std : [0.]})      
    test_error += test_error_
    y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:] = y_pred_test_
    y_true_test [ i * batch_size : (i + 1) * batch_size ,:] = y_true_test_
    y_h_true [ i * batch_size : (i + 1) * batch_size ,:] = X_h_ 
    y_h_pred [ i * batch_size : (i + 1) * batch_size ,:] = y_ae_h_       
test_error = (test_error/ avg_num) ** 0.5

#%% PRINTING COSTS
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#%%##########################################################################
# Savings network
AE_output={};
AE_output['y_true_test'] = y_true_test
AE_output['y_pred_test'] = y_pred_test
AE_output['input_dim'] = input_dim
AE_output['overlap'] = overlap
AE_output['y_h_true'] = y_h_true
AE_output['y_h_pred'] = y_h_pred

save_path = "/home/hsadeghi/Dropbox/july/conv_codec/highway_AE_output.mat"
sio.savemat(save_path, AE_output);
sess.close()