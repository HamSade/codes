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
from dcgan import regressor_noise, discriminator

from time import time
import matplotlib.pyplot as plt

#%% Hyper parameters

input_dim = 2 ** 13
z_dim = 100


batch_size = 32
display_step = 10
cost_lambda = 0.9


both_adam = False
apply_clip_weights =  True
clip_limit = 0.05

gpu_frac = 1.0
disc_update_cycles = 5
reg_update_cycles = 1

n_batch = 1
training_epochs = 1
num_training_files = 10

n_batch = 100
training_epochs = 16
num_training_files = 10 

learning_rate_init = 0.0002
lr_dec_fac = 1.
noise_std_init = [1.]
noise_dec_fac = 1.


#%% Printing

print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('input_dim', input_dim)

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

#x_in_reg = Sxx_l

# simple normal noise
noise = tf.random_uniform(tf.shape(Sxx_l), minval=-1.0, maxval=1.0)
#noise = tf.random_normal(tf.shape(Sxx_l), stddev = noise_std_init)
noise = tf.expand_dims (noise, axis=-1)


# upsampled uniform noise
#height = 129 
#width = input_dim // (2**7)
#noise = tf.random_uniform([batch_size, z_dim], minval=-1.0, maxval=1.0)
#noise = tf.layers.dense(noise, 1024 * width * height )
#noise = tf.reshape(noise,[-1, height, width, 1024])

ll = tf.expand_dims (Sxx_l, axis=-1)
x_in_reg = tf.concat([ll, noise], axis=-1)


with tf.variable_scope('reg'):
    reg_output_h = regressor_noise(x_in_reg, training = training)
    
reg_output = tf.concat([Sxx_l, reg_output_h], axis = 1, name = 'Sxx_reconstructed')

with tf.variable_scope('disc'):
    d_real = discriminator( Sxx_real, training = training)
#    d_real = discriminator( Sxx_h, training = training)
with tf.variable_scope('disc', reuse = True):
    d_fake = discriminator(reg_output, training = training)
#    d_fake = discriminator(reg_output_h, training = training)


#%%
def xentropy(logits, labels, name="x_entropy"):
        y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
#        tf.scalar_summary(name, xentropy)
        return y
#%% Cost definitions
# DCGAN
#disc_cost_dcgan =  1/2.* (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real, labels = tf.ones_like(d_real)))+
#        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.zeros_like(d_fake))) )
#reg_cost_dcgan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.ones_like(d_fake)))

# original GAN
#disc_cost_org = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
#reg_cost_org = -tf.reduce_mean(tf.log(d_fake))

##Wasserstein
#d_fake = tf.sigmoid(d_fake)
#d_real = tf.sigmoid(d_real)
disc_cost_w =  tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
reg_cost_w = -tf.reduce_mean(d_fake)

# xentropy
#disc_cost_real_x = xentropy(d_real, tf.ones_like(d_real))
#disc_cost_fake_x = xentropy(d_fake, tf.zeros_like(d_fake) )
#disc_cost_x = 0.5 * ( disc_cost_real_x + disc_cost_fake_x )
#reg_cost_x = xentropy(d_fake, tf.ones_like(d_fake) )

# Wasserstein
#reg_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.ones_like(d_real)))
#disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_real, tf.ones_like(d_real))) \
#            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.zeros_like(d_fake)))
    
## LSGAN (mean squared)
#disc_cost_real_mse = tf.reduce_mean( tf.squared_difference(d_real, 1.))
#disc_cost_fake_mse = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
#disc_cost_mse = 0.5 * (disc_cost_real_mse + disc_cost_fake_mse)
#reg_cost_mse = 0.5 * tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 

## LSGAN (mean squared)
#disc_cost_real_mse = tf.reduce_mean( tf.squared_difference(d_real, 1.))
#disc_cost_fake_mse = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
#disc_cost_mse = 0.5 * (disc_cost_real_mse + disc_cost_fake_mse)
#reg_cost_mse = 0.5 * tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 

#disc_cost = disc_cost_mse 
#reg_cost = reg_cost_mse + cost_lambda * reg_l1_cost
#reg_cost = reg_cost_mse + cost_lambda * reg_l2_cost


reg_l1_cost = tf.reduce_mean(tf.abs(Sxx_h - reg_output_h), name = 'reg_l1_cost')
#reg_l2_cost = tf.reduce_mean(tf.squared_difference(Sxx_h, reg_output_h), name = 'reg_l2_cost')

# Final cost
#disc_cost = disc_cost_mse 
#reg_cost = reg_cost_mse + cost_lambda * reg_l1_cost
#disc_cost = disc_cost_x
#reg_cost = reg_cost_x + cost_lambda * reg_l2_cost
#disc_cost = disc_cost_org
#reg_cost = reg_cost_org + cost_lambda * reg_l2_cost
#disc_cost = disc_cost_dcgan
#reg_cost = reg_cost_dcgan + cost_lambda * reg_l2_cost
disc_cost = disc_cost_w
reg_cost = reg_cost_w + cost_lambda * reg_l1_cost
#disc_cost = disc_cost_w + disc_cost_mse 
#reg_cost = reg_cost_w + gen_cost_mse
#disc_cost = disc_cost_mse 
#reg_cost = reg_cost_mse
#disc_cost = disc_cost_dcgan
#reg_cost = reg_cost_dcgan


#%% Variables
t_vars = tf.trainable_variables()
disc_vars = []
reg_vars = []
for var in t_vars:
    if var.name.startswith('disc'):
        disc_vars.append(var)
    if var.name.startswith('reg'):
        reg_vars.append(var)
for x in disc_vars:
#    print('disc_variable', x.name)
    assert x not in reg_vars
for x in reg_vars:
#    print('reg_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
#    print('all_variables', x.name)
    assert x in reg_vars or x in disc_vars, x.name
if apply_clip_weights:
    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')
    
#%% Optimizers
if both_adam:
    disc_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(disc_cost, var_list=disc_vars)
    reg_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(reg_cost, var_list=reg_vars)
    total_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2 = 0.999).minimize(reg_cost + disc_cost)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    reg_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(reg_cost, var_list=reg_vars)
    total_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(reg_cost + disc_cost)

#%%##############################################################################
# Training
#init = tf.global_variables_initializer()
#sess = tf.Session()
#sess.run(init)

init = tf.global_variables_initializer()

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
myconfig = tf.ConfigProto()   
myconfig.log_device_placement = False
myconfig.gpu_options.allow_growth=True    
sess = tf.Session(config=myconfig) #, gpu_options=gpu_options)

sess.run(init)

#%% Tensorboard setup
#tf.summary.scalar('reg_l2_cost', reg_l2_cost)
#tf.summary.scalar('reg_adv_cost', reg_adv_cost)
#tf.summary.scalar('disc_cost', disc_cost)
#merged_summary = tf.summary.merge_all()
#summary_writer = tf.summary.FileWriter('/home/hsadeghi/Dropbox/august/spectrogram/tensorboard/dc_BWE/1/')
#summary_writer.add_graph(sess.graph)
 
#%% Training cycle
start_time =  time()
counter = 0
for epoch in range(training_epochs):
    training_data = [[]]* num_training_files
    reading_phase = True
    for file_idx in range(num_training_files): 
        if reading_phase:
            training_data [file_idx] = data_loader(file_idx, input_dim)
        for  i in range(n_batch):
            
            counter += 1
            ################################################################
            # Update disc
            batch_xs = data_parser(training_data[file_idx], input_dim, batch_size)
#            with tf.device('/gpu:0'):
            for i in range(disc_update_cycles):
                _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                         feed_dict={L: batch_xs[0], H : batch_xs[1], maxi : batch_xs[2],
                                                    learning_rate: learning_rate_init,
                                                    noise_std : noise_std_init,
                                                    training : 1.})
    
            # clip disc weights
            if apply_clip_weights:
                sess.run(clip_disc_vars)
            
            ################################################################
            # Update regressor
            for j in range(reg_update_cycles):
                batch_xs = data_parser(training_data[file_idx], input_dim, batch_size)
                _, reg_cost_, reg_l1_cost_= sess.run([reg_opt, reg_cost, reg_l1_cost],
                                                     feed_dict={L: batch_xs[0],
                                                  H : batch_xs[1], maxi : batch_xs[2],
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init,
                                                        training : 1.})
    
#            _, reg_cost_= sess.run([reg_opt, reg_cost], feed_dict={L: batch_xs[0],
#                                                      H : batch_xs[1], maxi : batch_xs[2],
#                                                            learning_rate: learning_rate_init,
#                                                            noise_std : noise_std_init,
#                                                            training : 1.})

            time_lapsed = time()-start_time
#                summary_writer.add_summary(ss, i)  #tensorboard
            print("epoch:", '%02d' % (epoch + 1),
                    "File:", '%02d' % (file_idx),
                  "iteration:", '%04d' % (i + 1),
                  "disc_cost =", "{:.9f}".format( (10**4) * disc_cost_),
                  "reg_cost_ =", "{:.9f}".format( (10**4) * reg_cost_),
                  "reg_l1_cost =", "{:.9f}".format( (10**4) * reg_l1_cost_),
#                  "reg_l2_cost =", "{:.9f}".format( (10**4) * reg_l2_cost_),
                  'time lapsed =', '%02d' % (time_lapsed//3600), 'h:', '%02d'%((time_lapsed//60)%60),'m')
            
            if counter % (display_step) == 0:
                plt.figure(1)
                plt.plot(counter, (10**4) * disc_cost_, 'r*') 
                plt.hold(True)
                plt.pause(0.01)
                  
                plt.plot(counter, (10**4) * reg_cost_, 'b.') 
                plt.hold(True)
                plt.pause(0.01)
                
                plt.legend('Disc_cost','Gen_cost')
                
                #########display some sample spectrogram results
#                if i % display_step * 100 == 0:
                plt.figure(2)
                reg_output_ = sess.run([reg_output], feed_dict={L: batch_xs[0],
                                              H : batch_xs[1], maxi : batch_xs[2],
                                              noise_std : noise_std_init, training : 0.})
                plt.pcolormesh(reg_output_[0][8,:])
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.pause(0.01)
                
            
            ################################################################
            #### Updating both once in a while
#            if counter % (10*counter) == 0:
#                batch_xs = data_parser(training_data[file_idx], input_dim, batch_size)  # Hack #4
#                _ = sess.run([total_opt], feed_dict={L: batch_xs[0],
#                                              H : batch_xs[1], maxi : batch_xs[2],
#                                              learning_rate: learning_rate_init,
#                                              noise_std : noise_std_init,
#                                              training : 1.})
            
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
                                              H : sampled_x[1], maxi : batch_xs[2], noise_std : [0.],
                                                    training : 0. })
training_error = (training_error/ avg_num) ** 0.5

#%% Test error calculation
test_data = data_loader(10, input_dim)    
test_error = 0
avg_num = 50

#nfft = 128
#n_time_bins = input_dim//128
#y_pred_test = np.zeros([ avg_num*(batch_size), nfft+1       , n_time_bins])
#y_true_test = np.zeros([ avg_num*(batch_size), nfft+1       , n_time_bins])   
#y_h_true = np.zeros([avg_num*(batch_size),     nfft //2     , n_time_bins])
#y_h_pred = np.zeros([avg_num*(batch_size),     nfft //2     , n_time_bins])

y_p = []
y_t = []
maximum_spec = np.zeros(avg_num)

for i in range(avg_num):
    sampled_x = data_parser(test_data, input_dim, batch_size)  
#    Sxx_h_ , y_reg_h_,\
#    y_true_test_, y_pred_test_,\
#    test_error_, maxi_ = sess.run([Sxx_h , reg_output_h, Sxx_real, reg_output, reg_cost, maxi],\
#                                        feed_dict={L: sampled_x[0],
#                                              H : sampled_x[1], maxi : sampled_x[2],
#                                              noise_std : [0.],
#                                              training : 0.})      
    y_t_, y_p_,\
    test_error_, maxi_ = sess.run([Sxx_real , reg_output, reg_cost, maxi],\
                                        feed_dict={L: sampled_x[0],
                                              H : sampled_x[1], maxi : sampled_x[2],
                                              noise_std : [0.],
                                              training : 0.})
    
    test_error += test_error_
#    y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:,:] = y_pred_test_
#    y_true_test [ i * batch_size : (i + 1) * batch_size ,:,:] = y_true_test_
#    y_h_true [ i * batch_size : (i + 1) * batch_size ,:,:] = Sxx_h_ 
#    y_h_pred [ i * batch_size : (i + 1) * batch_size ,:,:] = y_reg_h_ 

    if y_t == []:
        y_t = y_t_
        y_p = y_p_
    else:
        y_t = np.concatenate((y_t, y_t_))
        y_p = np.concatenate((y_p, y_p_))

    maximum_spec[i] = maxi_
      
test_error = (test_error/ avg_num) ** 0.5

#%% PRINTING COSTS
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#%%##########################################################################
# Savings network
AE_output={};
#AE_output['y_true_test'] = y_true_test
#AE_output['y_pred_test'] = y_pred_test

AE_output['y_true_test'] = y_t
AE_output['y_pred_test'] = y_p


AE_output['input_dim'] = input_dim
#AE_output['y_h_true'] = y_h_true
#AE_output['y_h_pred'] = y_h_pred
AE_output['maximum_spec'] = maximum_spec

save_path = "/home/hsadeghi/Dropbox/september/spectrogram_bwe/dcgan_BWE_output.mat"
sio.savemat(save_path, AE_output);
sess.close()