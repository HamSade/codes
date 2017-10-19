#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:04:27 2017

@author: hsadeghi
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import scipy.io as sio 

from spectrogram_data_loader import data_loader, data_parser
from dcgan import regressor #, discriminator

#%% Main parameters

input_dim = 2 ** 11
gpu_frac = 1.0

n_batch = 1
training_epochs = 1
num_training_files = 10
#
n_batch = 100
training_epochs = 16
num_training_files = 10 

learning_rate_init = 0.0001
lr_dec_fac = 1.
noise_std_init = [0.0]
noise_dec_fac = 1.


#%% Other Parameters
batch_size = 128
display_step = 10

print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('input_dim', input_dim)
print('batch_size', batch_size)

#%% ##############################################################################
L = tf.placeholder("float", [batch_size, None, None], name='low_input')
H = tf.placeholder("float", [batch_size, None, None], name='high_input')
#drop_out_p=tf.placeholder("float", [1])
training = tf.placeholder("float",None, name = 'training_indicator')
learning_rate = tf.placeholder("float", None, name = 'learning_rate') 
noise_std = tf.placeholder("float", 1, name = 'noise_std')     
maxi = tf.placeholder("float", None, name = 'maximum_spec_value')

#%%##############################################################################
# Building the model
Sxx_l = L
Sxx_h = H

Sxx_real = tf.concat([Sxx_l, Sxx_h], axis = 1, name='Sxx_real')

#noise = tf.random_normal(shape = X_l_recon.get_shape().as_list(), stddev = noise_std)
#x_in_highway = (X_l, noise)
x_in_highway = Sxx_l

with tf.variable_scope('reg'):
    reg_output_h = regressor(x_in_highway, training = training)

reg_output = tf.concat([Sxx_l, reg_output_h], axis = 1, name = 'Sxx_reconstructed')

#%% Cost definitions
#reg_cost = tf.reduce_mean(tf.squared_difference(Sxx_real, reg_output)) 
reg_cost = tf.reduce_mean(tf.squared_difference(Sxx_h, reg_output_h), name = 'reg_cost') 

#%% Optimizers
opt = tf.train.AdamOptimizer(learning_rate).minimize(reg_cost)

#%%##############################################################################
# Training
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, gpu_options=gpu_options))
sess.run(init)

#%% Tensorboard setup
tf.summary.scalar('cost', reg_cost)
merged_summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/home/hsadeghi/Dropbox/august/spectrogram/tensorboard/dc_BWE/1/')
summary_writer.add_graph(sess.graph)
 
#%% Training cycle
for epoch in range(training_epochs):
    training_data = [[]]* num_training_files
    reading_phase = True
    for file_idx in range(num_training_files): 
        if reading_phase:
            training_data [file_idx] = data_loader(file_idx, input_dim)
        for  i in range(n_batch):
            #pre_emph is zero since we do it on X if we want separatrely            
            batch_xs = data_parser(training_data[file_idx], input_dim, batch_size)         
            ################################################################
            # Update regressor
            _, reg_cost_, ss= sess.run([opt, reg_cost, merged_summary], feed_dict={L: batch_xs[0],
                                              H : batch_xs[1],
                                                    learning_rate: learning_rate_init,
                                                    noise_std : noise_std_init,
                                                    training : 1.})
#####       Display logs per epoch step
            if i % display_step == 0:
                summary_writer.add_summary(ss, i)  #tensorboard
                print("epoch:", '%02d' % (epoch + 1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i + 1),
                      "reg_cost =", "{:.9f}".format( (10**4) * reg_cost_))
            ################################################################3
    
        learning_rate_init = max( 0.0001, learning_rate_init /lr_dec_fac)
        noise_std_init[0] =   max(0.01, noise_std_init[0]/ noise_dec_fac  )
    reading_phase = False
print("Optimization Finished!")

#%%##########################################################################
# Training error calculation
training_data= training_data[9] #data_loader(9, input_dim)
training_error = 0
avg_num = 50
for i in range(avg_num):
    sampled_x = data_parser(training_data, input_dim, batch_size)
    training_error += sess.run(reg_cost, feed_dict={L: sampled_x[0],
                                              H : sampled_x[1], noise_std : [0.],
                                                    training : 0. })
training_error = (training_error/ avg_num) ** 0.5

#%% Test error calculation
test_data = data_loader(10, input_dim)    
test_error = 0
avg_num = 50

nfft = 128
n_time_bins = input_dim//128
y_pred_test = np.zeros([ avg_num*(batch_size), nfft+1       , n_time_bins])
y_true_test = np.zeros([ avg_num*(batch_size), nfft+1       , n_time_bins])   
y_h_true = np.zeros([avg_num*(batch_size),     nfft //2     , n_time_bins])
y_h_pred = np.zeros([avg_num*(batch_size),     nfft //2     , n_time_bins])
maximum_spec = np.zeros(avg_num)

for i in range(avg_num):
    sampled_x = data_parser(test_data, input_dim, batch_size)  
    Sxx_h_ , y_reg_h_,\
    y_true_test_, y_pred_test_,\
    test_error_, maxi_ = sess.run([Sxx_h , reg_output_h, Sxx_real, reg_output, reg_cost, maxi],\
                                        feed_dict={L: sampled_x[0],
                                              H : sampled_x[1], maxi : sampled_x[2],
                                              noise_std : [0.],
                                              training : 0.})      
    test_error += test_error_
    y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:,:] = y_pred_test_
    y_true_test [ i * batch_size : (i + 1) * batch_size ,:,:] = y_true_test_
    y_h_true [ i * batch_size : (i + 1) * batch_size ,:,:] = Sxx_h_ 
    y_h_pred [ i * batch_size : (i + 1) * batch_size ,:,:] = y_reg_h_ 
    maximum_spec[i] = maxi_
      
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
AE_output['y_h_true'] = y_h_true
AE_output['y_h_pred'] = y_h_pred
AE_output['maximum_spec'] = maximum_spec

save_path = "/home/hsadeghi/Dropbox/august/spectrogram/dc_BWE_output.mat"
sio.savemat(save_path, AE_output);
sess.close()