#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:04:07 2017

@author: hsadeghi
"""

from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import scipy.io as sio 

from spectrogram_data_loader import data_loader, data_parser
from dcgan import generator, discriminator

#%% Main parameters

input_dim = 2 ** 15
nfft = 128

both_adam = True
apply_clip_weights =  False
clip_limit = 0.9
cost_lambda = 0.

gpu_frac = 1.0
disc_update_cycles = 1

n_batch = 1
training_epochs = 1
num_training_files = 10

n_batch = 100
training_epochs = 3
num_training_files = 10 

learning_rate_init = 0.01
lr_dec_fac = 2
noise_std_init = [0.]
noise_dec_fac = 1.


#%% Other Parameters
batch_size = 128
display_step = 10

print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('input_dim', input_dim)

#%% ##############################################################################
X = tf.placeholder("float", [batch_size, None, None], name='spectrogram_input')
training = tf.placeholder("float",None, name = 'training_indicator')
learning_rate = tf.placeholder("float", None, name = 'learning_rate') 
noise_std = tf.placeholder("float", 1, name = 'noise_std')     
maxi = tf.placeholder("float", None, name = 'maximum_spec_value')

#%%##############################################################################
# Building the model
Sxx = X

Sxx_real = Sxx
#x_in_gen = Sxx.get_shape().as_list()
x_in_gen = [batch_size, nfft, input_dim]

with tf.variable_scope('gen'):
    gen_output = generator(x_in_gen, training = training)

with tf.variable_scope('disc'):
    d_real = discriminator( Sxx_real, training = training)
with tf.variable_scope('disc', reuse = True):
    d_fake = discriminator(gen_output, training = training)

#%% Cost definitions
## mean squared
disc_cost_real = tf.reduce_mean( tf.squared_difference(d_real, 1.))
disc_cost_fake = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
# log norm
#disc_cost_real =  - tf.reduce_mean( tf.log(d_real))
#disc_cost_fake = tf.reduce_mean( tf.log( 1. - d_fake))
disc_cost = disc_cost_real + disc_cost_fake

#gen_adv_cost =  - tf.reduce_mean(tf.log(d_fake))
gen_adv_cost = tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 
#gen_l2_cost = tf.reduce_mean(tf.squared_difference(Sxx_h, gen_output_h), name = 'gen_cost')
gen_cost = gen_adv_cost 
#gen_cost = gen_adv_cost + cost_lambda * gen_l2_cost

#%% Variables
t_vars = tf.trainable_variables()
disc_vars = []
gen_vars = []
for var in t_vars:
    if var.name.startswith('disc'):
        disc_vars.append(var)
    if var.name.startswith('gen'):
        gen_vars.append(var)
for x in disc_vars:
    print('disc_variable', x.name)
    assert x not in gen_vars
for x in gen_vars:
    print('gen_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
    print('all_variables', x.name)
    assert x in gen_vars or x in disc_vars, x.name
if apply_clip_weights:
    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')
    
#%% Optimizers
if both_adam:
    disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_cost, var_list=gen_vars)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_cost, var_list=gen_vars)

#%%##############################################################################
# Training
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, gpu_options=gpu_options))
sess.run(init)

#%% Tensorboard setup
#tf.summary.scalar('gen_l2_cost', gen_l2_cost)
#tf.summary.scalar('gen_adv_cost', gen_adv_cost)
#tf.summary.scalar('disc_cost', disc_cost)
#merged_summary = tf.summary.merge_all()
#summary_writer = tf.summary.FileWriter('/home/hsadeghi/Dropbox/august/spectrogram/tensorboard/dc_BWE/1/')
#summary_writer.add_graph(sess.graph)
 
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
            # Update disc
            with tf.device('/gpu:0'):
                for i in range(disc_update_cycles):
                    _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                             feed_dict={X: batch_xs[0], maxi : batch_xs[1],
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init,
                                                        training : 1.})
#                     Display logs per epoch step
#                    if i % display_step == 0:
#                        print("Discriminator_cost =", "{:.9f}".format( (10**4) * disc_cost_))

                # clip disc weights
                if apply_clip_weights:
                    sess.run(clip_disc_vars)
            
            ################################################################
            # Update genressor
            with tf.device('/gpu:1'):
                _, gen_cost_ = sess.run([gen_opt, gen_cost], feed_dict={X: batch_xs[0],
                                                  maxi : batch_xs[1],
                                                  learning_rate: learning_rate_init,
                                                  noise_std : noise_std_init,
                                                  training : 1.})
#            if gen_cost_ < 5010:
#                learning_rate_init = learning_rate_init/10
#           Display logs per epoch step
            if i % display_step == 0:
#                summary_writer.add_summary(ss, i)  #tensorboard
                print("epoch:", '%02d' % (epoch + 1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i + 1),
                      "disc_cost =", "{:.9f}".format( (10**4) * disc_cost_),
                      "gen_adv_cost =", "{:.9f}".format( (10**4) * gen_cost_))
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
    training_error += sess.run(gen_cost, feed_dict={X: sampled_x[0],
                                              maxi : batch_xs[1], noise_std : [0.],
                                                    training : 0. })
training_error = (training_error/ avg_num) ** 0.5

#%% Test error calculation
test_data = data_loader(10, input_dim)    
test_error = 0
avg_num = 50

#n_time_bins = 64
#nfft = 128

y_p = [] #np.zeros([ avg_num*(batch_size), nfft, n_time_bins])
y_t = [] # np.zeros([ avg_num*(batch_size), nfft, n_time_bins])   
maximum_spec = np.zeros(avg_num)

for i in range(avg_num):
    sampled_x = data_parser(test_data, input_dim, batch_size)  
    y_true_test_, y_pred_test_,\
    test_error_, maxi_ = sess.run([Sxx_real , gen_output, gen_cost, maxi],\
                                        feed_dict={X: sampled_x[0],
                                              maxi : sampled_x[1],
                                              noise_std : [0.],
                                              training : 0.})      
    test_error += test_error_
    if y_p == []:
        y_p = y_pred_test_
        y_t = y_true_test_
    else:
        y_p = np.concatenate((y_p, y_pred_test_))
        y_t = np.concatenate((y_t, y_true_test_))
                 

    maximum_spec[i] = maxi_
      
test_error = (test_error/ avg_num) ** 0.5
#    y_p [ i * batch_size : (i + 1) * batch_size  ,:,:] = y_pred_test_
#    y_t [ i * batch_size : (i + 1) * batch_size ,:,:] = y_true_test_ 

#%% PRINTING COSTS
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#%%##########################################################################
# Savings network
AE_output={};
AE_output['y_true_test'] = y_t
AE_output['y_pred_test'] = y_p
AE_output['input_dim'] = input_dim
AE_output['maximum_spec'] = maximum_spec

save_path = "/home/hsadeghi/Dropbox/august/spectrogram_generation/gan_dc_spec_gen_output.mat"
sio.savemat(save_path, AE_output);
sess.close()