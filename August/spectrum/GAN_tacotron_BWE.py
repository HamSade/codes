#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:07:00 2017

@author: hsadeghi
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as sio 
from data_loader import data_loader, data_parser
#from conv_GAN import discriminator
#from taco_ops import deep_highway_with_carrier #, deep_taco

from tacotron import taco_BWE, highway_disc

#%% Main parameters

input_dim = 2 ** 10
overlap= int(input_dim/2**4)

gpu_frac = 1.0

apply_mask = True # Whether to input zero-padded test input
apply_clip_weights = True
both_adam =  True
disc_update_cycles = 1
clip_limit = 0.05


cost_lambda = 1.0


num_layers_disc = 3
num_conv = 2
num_highway = 2

n_batch = 100
training_epochs = 8
num_training_files = 10

#n_batch = 1
#training_epochs = 1
#num_training_files = 10


learning_rate_init = 0.0001
lr_dec_fac = 2
noise_std_init = [0.01]
noise_dec_fac = 1.


#%% Other Parameters
batch_size = 128
display_step = 100

dropout_p = 1.

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
x_in_taco = (X_l, noise)

with tf.variable_scope('taco'):
    taco_output_h = taco_BWE(x_in_taco, num_conv = num_conv, num_highway = num_highway)

taco_output = X_l + taco_output_h

noise_h_real = tf.random_normal(shape = X_h.get_shape().as_list(), stddev = noise_std)
noise_h_taco = tf.random_normal(shape = taco_output_h.get_shape().as_list(), stddev = noise_std)

with tf.variable_scope('disc'):
    d_real = highway_disc(tf.multiply( X_h, noise_h_real), num_layers_disc)
#    d_real = discriminator(x_real, comp_ratio_disc)
with tf.variable_scope('disc', reuse=True):
#    d_fake = discriminator(taco_output_h, comp_ratio_disc)
    d_fake = highway_disc(tf.multiply( taco_output_h, noise_h_taco), num_layers_disc)
#    d_fake = discriminator(taco_output, comp_ratio_disc)
    
    
#%% Cost definitions
# mean squared
disc_cost_real = tf.reduce_mean( tf.squared_difference(d_real, 1.))
disc_cost_fake = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
# log norm
#disc_cost_real =  - tf.reduce_mean( tf.log(d_real))
#disc_cost_fake = tf.reduce_mean( tf.log( 1. - d_fake))
disc_cost = disc_cost_real + disc_cost_fake

#taco_adv_cost =  - tf.reduce_mean(tf.log(d_fake))
#taco_cost = tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 
taco_adv_cost = tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 
#taco_cost = taco_adv_cost + cost_lambda * tf.reduce_mean(tf.abs(tf.subtract(x_real, taco_output)))
# Considering only highpass speech
#taco_cost = taco_adv_cost + cost_lambda * tf.reduce_mean(tf.abs(tf.subtract(X_h, taco_output_h)))
#taco_l1_cost = tf.reduce_mean(tf.abs(X_h - taco_output_h))
taco_l2_cost = tf.reduce_mean(tf.squared_difference(X_h, taco_output_h))
taco_cost = taco_adv_cost + cost_lambda * taco_l2_cost

#%% Variables
t_vars = tf.trainable_variables()
disc_vars = []
taco_vars = []
for var in t_vars:
    if var.name.startswith('disc'):
        disc_vars.append(var)
    if var.name.startswith('taco'):
        taco_vars.append(var)
for x in disc_vars:
    print('disc_variable', x.name)
    assert x not in taco_vars
for x in taco_vars:
    print('taco_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
    print('all_variables', x.name)
    assert x in taco_vars or x in disc_vars, x.name
if apply_clip_weights:
    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')

#%% Optimizers

#taco_opt = tf.train.AdamOptimizer(learning_rate).minimize(taco_cost, var_list=taco_vars)
#disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
if both_adam:
    disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    taco_opt = tf.train.AdamOptimizer(learning_rate).minimize(taco_cost, var_list=taco_vars)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    taco_opt = tf.train.AdamOptimizer(learning_rate).minimize(taco_cost, var_list=taco_vars)

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
#count = 0
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
            
            ################################################################
            # Update disc
            with tf.device('/gpu:0'):
                for i in range(disc_update_cycles):
                    _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                             feed_dict={X: batch_xs,
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init})
#                     Display logs per epoch step
#                    if i % display_step == 0:
#                        print("Discriminator_cost =", "{:.9f}".format( (10**4) * disc_cost_))
#                
                
                # clip disc weights
                if apply_clip_weights:
                    sess.run(clip_disc_vars)
            ################################################################
            # Update AE
            with tf.device('/gpu:1'):
                _, taco_l2_cost_, taco_adv_cost_ = sess.run([taco_opt, taco_l2_cost,taco_adv_cost],
                                       feed_dict={X: batch_xs,
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init})
#####             Display logs per epoch step
                if i % display_step == 0:
                    print("epoch:", '%02d' % (epoch + 1),
                            "File:", '%02d' % (file_idx),
                          "iteration:", '%04d' % (i + 1),
                          "Discriminator_cost =", "{:.9f}".format( (10**4) * disc_cost_),#   "taco_cost =", "{:.9f}".format( (10**4) * taco_cost_),
                          "taco_l2_cost =", "{:.9f}".format( (10**4) * taco_l2_cost_ ),
                          "taco_adv_cost", "{:.9f}".format( (10**4) * taco_adv_cost_) )
                    
#               ###  Early stopping!
#                if taco_l2_cost_ < 500/10000:
#                    print('limit reached')
#                    if count == 0:
#                        learning_rate_init = learning_rate_init/10
#                        count += 1

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
    sampled_x = data_parser(training_data, input_dim, batch_size, overlap=overlap)
    training_error += sess.run(taco_l2_cost, feed_dict={X: sampled_x, noise_std : [0.]})
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
    X_h_ , y_taco_h_,\
    y_true_test_, y_pred_test_,\
    test_error_ = sess.run([X_h , taco_output_h, x_real, taco_output, taco_l2_cost],\
                                        feed_dict={X: sampled_x, noise_std : [0.]})      
    test_error += test_error_
    y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:] = y_pred_test_
    y_true_test [ i * batch_size : (i + 1) * batch_size ,:] = y_true_test_
    y_h_true [ i * batch_size : (i + 1) * batch_size ,:] = X_h_ 
    y_h_pred [ i * batch_size : (i + 1) * batch_size ,:] = y_taco_h_ 
      
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

save_path = "/home/hsadeghi/Dropbox/august/GAN_taco_output.mat"
sio.savemat(save_path, AE_output);
sess.close()