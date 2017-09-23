#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:51:12 2017

@author: hsadeghi
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as sio 
from data_loader import data_loader, data_parser
from conv_GAN import discriminator
from highway_ops import deep_highway_with_carrier, deep_highway

#%% Main parameters

input_dim = 2 ** 10
overlap= int(input_dim/2**4)

gpu_frac = 1.0

apply_mask = True # Whether to input zero-padded test input
apply_clip_weights = True
both_adam =  True
disc_update_cycles = 1
clip_limit = 0.05

layer_widths = [input_dim + 2 * overlap] * 30
cost_lambda = 0.1
comp_ratio_disc = 2 ** 6

n_batch = 100
training_epochs = 1
num_training_files = 10

#n_batch = 1
#training_epochs = 1
#num_training_files = 10


learning_rate_init = 0.01
lr_dec_fac = 1.4
noise_std_init = [0.01]
noise_dec_fac = np.sqrt(2)

#%% Other Parameters
batch_size = 128
display_step = 100

dropout_p = 1.

print('comp_ratio_disc', comp_ratio_disc)
print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('cost_lambda', cost_lambda)
print('overlap', overlap)
print('input_dim', input_dim)
print('batch_size', batch_size)

#%% Rectanular mask buildup 
#zeros = np.zeros(int(overlap))
#flat = np.ones(input_dim)
#mask = np.concatenate([zeros, flat, zeros])   

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


noise = tf.random_normal(shape = X_l_recon.get_shape().as_list(), stddev = noise_std)
x_in_highway = (X_l_recon, noise)
#x_in_highway = X_l_recon

with tf.variable_scope('highway'):
    highway_output_h = deep_highway_with_carrier(x_in_highway, layer_widths)
#    highway_output_h = deep_highway(x_in_highway, layer_widths)

highway_output = X_l + highway_output_h
#highway_output = highway_output_h

with tf.variable_scope('disc'):
#    d_real = discriminator(X_h, comp_ratio_disc)  # Disc onlu judges highpass signal
    d_real = discriminator(x_real, comp_ratio_disc)
with tf.variable_scope('disc', reuse=True):
    d_fake = discriminator(highway_output, comp_ratio_disc)

#%% Cost definitions
# mean squared
disc_cost_real = tf.reduce_mean( tf.squared_difference(d_real, 1.))
disc_cost_fake = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
### Wasserstein
#disc_cost_real = tf.reduce_max( tf.abs( d_real - 1.))
#disc_cost_fake = tf.reduce_max( tf.abs( d_fake))
# L1 norm
#disc_cost_real = tf.reduce_mean( tf.abs(d_real - 1.))
#disc_cost_fake = tf.reduce_mean( tf.abs(d_fake))
disc_cost = disc_cost_real + disc_cost_fake

highway_cost = tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 
#highway_cost = tf.reduce_mean(tf.abs(d_fake - 1.)) 
#highway_cost = tf.reduce_max( tf.abs(d_fake - 1.)) 
#highway_adv_cost = tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 
#highway_cost = highway_adv_cost + cost_lambda * tf.reduce_mean(tf.abs(tf.subtract(x_real, highway_output)))
# Considering only highpass speech
#highway_cost = highway_adv_cost + cost_lambda * tf.reduce_mean(tf.abs(tf.subtract(X_h, highway_output_h)))
#highway_cost = highway_adv_cost + cost_lambda * tf.reduce_mean(tf.squared_difference(X_h, highway_output_h))

#%% Variables
t_vars = tf.trainable_variables()
disc_vars = []
highway_vars = []
for var in t_vars:
    if var.name.startswith('disc'):
        disc_vars.append(var)
    if var.name.startswith('highway'):
        highway_vars.append(var)
for x in disc_vars:
    print('disc_variable', x.name)
    assert x not in highway_vars
for x in highway_vars:
    print('highway_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
    print('all_variables', x.name)
    assert x in highway_vars or x in disc_vars, x.name
if apply_clip_weights:
    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')

#%% Optimizers

#highway_opt = tf.train.AdamOptimizer(learning_rate).minimize(highway_cost, var_list=highway_vars)
#disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
if both_adam:
    disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    highway_opt = tf.train.AdamOptimizer(learning_rate).minimize(highway_cost, var_list=highway_vars)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    highway_opt = tf.train.AdamOptimizer(learning_rate).minimize(highway_cost, var_list=highway_vars)

#%%##############################################################################
# Training
init = tf.global_variables_initializer()
#sess=tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, gpu_options=gpu_options))
# NO GPU
#config = tf.ConfigProto(device_count = {'GPU': 0})
#sess = tf.Session(config=config)
sess.run(init)

#%% Training cycle
for epoch in range(training_epochs):
    training_data = [[]]* num_training_files
    reading_phase = True
    for file_idx in range(num_training_files): 
        if reading_phase:
            training_data [file_idx] = data_loader(file_idx, input_dim)
        for  i in range(n_batch):
            #pre_emph is zero since we do it on X if we want separatrely            
            batch_xs = data_parser(training_data[file_idx], input_dim, batch_size,
                                   overlap = overlap)         
            # Update disc
            with tf.device('/gpu:0'):
                for i in range(disc_update_cycles):
                    _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                             feed_dict={X: batch_xs,
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init})
                    # Display logs per epoch step
#                    if i % display_step == 0:
#                        print("Discriminator_cost =", "{:.9f}".format( (10**4) * disc_cost_))
#                
                
                # clip disc weights
                if apply_clip_weights:
                    sess.run(clip_disc_vars) 
            # Update AE
            with tf.device('/gpu:1'):
                _, highway_cost_ = sess.run([highway_opt, highway_cost],
                                       feed_dict={X: batch_xs,
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init})
            # Display logs per epoch step
                if i % display_step == 0:
                    print("epoch:", '%02d' % (epoch + 1),
                            "File:", '%02d' % (file_idx),
                          "iteration:", '%04d' % (i + 1),
                          "Discriminator_cost =", "{:.9f}".format( (10**4) * disc_cost_),
                          "highway_cost =", "{:.9f}".format( (10**4) * highway_cost_)) 
                    
                # Early stopping!
#                if highway_cost_ < 1000/10000:
#                    print('limit reached')
#                    learning_rate_init = learning_rate_init/10
                        

        learning_rate_init = max( 0.00001, learning_rate_init /lr_dec_fac)
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
    training_error += sess.run(highway_cost, feed_dict={X: sampled_x, noise_std : [0.]})
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
    X_h_ , y_highway_h_,\
    y_true_test_, y_pred_test_,\
    test_error_ = sess.run([X_h , highway_output_h, x_real, highway_output, highway_cost],\
                                        feed_dict={X: sampled_x, noise_std : [0.]})      
    test_error += test_error_
    y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:] = y_pred_test_
    y_true_test [ i * batch_size : (i + 1) * batch_size ,:] = y_true_test_
    y_h_true [ i * batch_size : (i + 1) * batch_size ,:] = X_h_ 
    y_h_pred [ i * batch_size : (i + 1) * batch_size ,:] = y_highway_h_ 
      
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

save_path = "/home/hsadeghi/Dropbox/july/conv_codec/GAN_highway_output.mat"
sio.savemat(save_path, AE_output);
sess.close()