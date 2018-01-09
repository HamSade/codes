#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:24:07 2017

@author: hsadeghi
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import scipy.io as sio 

from spectrogram_loader import data_loader, data_parser
from model_coder import Coder

from time import time
import matplotlib.pyplot as plt

#%% Hyper parameters

input_dim = 2 ** 13
batch_size = 128
down_ratio = 4

num_rows = 257 // down_ratio + 1
num_cols = 64 // down_ratio

display_step = 10
cost_lambda = 10 # Gulrajani
cost_gamma = 0.9 # Face-super resolution

coder_adam = True
apply_clip_weights =  True
clip_limit = 0.01

gpu_frac = 0.9
disc_update_cycles = 1
coder_update_cycles = 1

#n_batch = 1
#training_epochs = 1
#num_training_files = 1
#
n_batch = 100
training_epochs = 16
num_training_files = 10 

learning_rate_init = 0.0002
lr_dec_fac = 1.
noise_std_init = [1.0]
noise_dec_fac = 1.0

noise_std_min = 0.01

#%% Printing
print('input_dim = ', input_dim)
print('batch_size = ', batch_size)
print('down_ratio = ', down_ratio)
print('num_rows = ', num_rows)
print('num_cols = ', num_cols)
print('cost_lambda = ',cost_lambda )
print('cost_gamma = ', cost_gamma)
print('clip_limit = ', clip_limit)
print('training_epochs = ', training_epochs)
print('learning_rate_init = ', learning_rate_init)
print('noise_std_init = ', noise_std_init)
print('noise_std_min = ', noise_std_min)

#%% PLACE HOLDERS
#batch_size = tf.placeholder(tf.int32, None, name = 'batch_size')
X = tf.placeholder("float", [batch_size, num_rows, num_cols], name='input')
training = tf.placeholder("float", None, name = 'training_indicator')
learning_rate = tf.placeholder("float", None, name = 'learning_rate') 
noise_std = tf.placeholder("float", [1], name = 'noise_std')     
maxi = tf.placeholder("float", None, name = 'maximum_spec_value')

#%%##############################################################################
# Building the model
# noise #noise = tf.random_uniform(tf.shape(Sxx_l), minval=-1.0, maxval=1.0)
#noise = tf.random_normal(tf.shape(Sxx_l), stddev = noise_std_init)
#noise = tf.expand_dims (noise, axis=-1)
coder = Coder(batch_size)
coder_output, d_real, d_fake, bits = coder(X[:, :-1, :], noise_std=noise_std, training = training)
tf.identity(coder_output, name="coder_output")
tf.identity(bits, name="bits_quantizer")

#%% Cost definition
#disc_cost =  tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
#coder_cost_adv = -tf.reduce_mean(d_fake)
#disc_cost =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.ones([batch_size], dtype=tf.int64), logits=d_real))+\
#                    tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.zeros([batch_size], dtype=tf.int64), logits=d_fake))
#                    
#coder_cost_adv = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(
#                    labels=tf.ones([batch_size], dtype=tf.int64), logits=d_fake))

disc_cost = tf.reduce_mean(tf.square(d_real - 1.) + tf.square(d_fake)) / 2. 
coder_cost_adv = tf.reduce_mean(tf.square(d_fake - 1.0)) / 2.


coder_l1_cost = tf.reduce_mean(tf.reduce_mean( tf.abs(X[:,:-1,:] - coder_output), axis=[-1,-2]))
#coder_cost = (1.0 - cost_lambda) * coder_cost_adv + cost_lambda * coder_l1_cost
coder_cost =  (1.0 - cost_gamma) * coder_cost_adv + cost_gamma * coder_l1_cost


#%% 'WGAN-GP':
# clip_disc_weights = None ===> means apply_clip_weights =  False

#alpha = tf.random_uniform( shape=[batch_size,1], minval=0., maxval=1.)
#differences = d_fake - d_real
#interpolates = d_real + (alpha*differences)
#
#training_bool = tf.less(training, 0.5)
#gradients = tf.gradients(coder.disc(interpolates, training=training_bool), [interpolates])[0]
#slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
#gradient_penalty = tf.reduce_mean((slopes-1.)**2)
#disc_cost += cost_lambda * gradient_penalty

#%% Variables
t_vars = tf.trainable_variables()
disc_vars = []
coder_vars = []
for var in t_vars:
    if var.name.startswith('disc'):
        disc_vars.append(var)
    if var.name.startswith('coder'):
        coder_vars.append(var)
for x in disc_vars:
#    print('disc_variable', x.name)
    assert x not in coder_vars
for x in coder_vars:
#    print('coder_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
#    print('all_variables', x.name)
    assert x in coder_vars or x in disc_vars, x.name
if apply_clip_weights:
    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')
    
#%% Optimizers
if coder_adam:
#    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    disc_opt =  tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.9).minimize(disc_cost, var_list=disc_vars)
    coder_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.9).minimize(coder_cost, var_list=coder_vars)
#    total_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.9).minimize(coder_cost + disc_cost)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list=disc_vars)
    coder_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(coder_cost, var_list=coder_vars)
    
#%% Session buildup 
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
myconfig = tf.ConfigProto()   
myconfig.log_device_placement = False
myconfig.gpu_options.allow_growth = True   

sess = tf.Session(config=myconfig) #, gpu_options=gpu_options)
sess.run(init)

#%% Training cycle

print('*'*100)
print('*'*100)
print('*'*100)
print('*'*50 + ' '*15+ 'TRAINING STARTED')
print('*'*100)
print('*'*100)

cdr_vec = []
disc_vec = []
l1_vec = []

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
            ################################################################
#            with tf.device('/gpu:0'):
            for j in range(disc_update_cycles):
                batch_xs = data_parser(training_data[file_idx], input_dim, batch_size, down_ratio)
                _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                         feed_dict={X: batch_xs[0], maxi : batch_xs[1],
                                                    learning_rate: learning_rate_init,
                                                    noise_std : noise_std_init,
                                                    training : 0.})
    
            disc_vec += [disc_cost_]
    
            # clip disc weights
            if apply_clip_weights:
                sess.run(clip_disc_vars)
            
            ################################################################
            # Update coder
            ################################################################
            for j in range(coder_update_cycles):
                batch_xs = data_parser(training_data[file_idx], input_dim, batch_size, down_ratio)
                _, coder_cost_adv_, coder_l1_cost_= sess.run([coder_opt, coder_cost_adv, coder_l1_cost],
                                                     feed_dict={X: batch_xs[0],
                                                        maxi : batch_xs[1],
                                                        learning_rate: learning_rate_init,
                                                        noise_std : noise_std_init,
                                                        training : 0.})
            cdr_vec += [coder_cost_adv_]
            l1_vec += [coder_l1_cost_]

            time_lapsed = time()-start_time
#                summary_writer.add_summary(ss, i)  #tensorboard
            print("epoch:", '%02d' % (epoch + 1),
                    "File:", '%02d' % (file_idx),
                  "iteration:", '%04d' % (i + 1),
                  "disc_cost =", "{:.3f}".format( (10**4) * disc_cost_),
                  "coder_cost_adv =", "{:.3f}".format( (10**4) * coder_cost_adv_),
                  "coder_l1_cost =", "{:.3f}".format( (10**4) * coder_l1_cost_),
                  'time lapsed =', '%02d' % (time_lapsed//3600), 'h:', '%02d'%((time_lapsed//60)%60),'m')
            
            if counter % (display_step) == 0:
                plt.figure(1)
                
                if np.abs(disc_cost_)<1.5:
                    plt.plot(counter, (10**4) * disc_cost_, 'r*') 
                    plt.hold(True)
                    plt.pause(0.01)
                  
                if np.abs(coder_cost_adv_)<1.5:
                    plt.plot(counter, (10**4) * coder_cost_adv_, 'b.') 
                    plt.hold(True)
                    plt.pause(0.01)
                
                if np.abs(coder_l1_cost_)<1.5:
                    plt.plot(counter, (10**4) * coder_l1_cost_, 'k.') 
                    plt.hold(True)
                    plt.pause(0.01)
                
#                plt.legend('Disc_cost','Gen_cost')
                
                #########display some sample spectrogram results
#                if i % display_step * 100 == 0:
                plt.figure(2)
                coder_output_ = sess.run([coder_output], feed_dict={X: batch_xs[0],
                                              maxi : batch_xs[1],
                                              noise_std : noise_std_init, training : 1.})
                plt.pcolormesh(coder_output_[0][0,:])   # frame number 8th in the batch
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
        noise_std_init[0] =   max(noise_std_min, noise_std_init[0]/ noise_dec_fac  )
    reading_phase = False
    
print('*'*50)
print('*'*50)
print("********************  TRAINING FINISHED!")
print('*'*50)
print('*'*50)
#%% Saving model
saver = tf.train.Saver()
save_path = saver.save(sess, '/vol/grid-solar/sgeusers/hsadeghi/research_results/saved_model/model.ckpt')
#saver.save(sess, 'my_test_model')
print("Model saved in file: %s" % save_path)

#%% Writing costs
training_costs={};    
training_costs['l1_cost'] = l1_vec
training_costs['coder_cost'] = cdr_vec
training_costs['disc_cost'] = disc_vec

#save_path = "events_training.mat"
save_path = "/vol/grid-solar/sgeusers/hsadeghi/research_results/saved_model/events_training.mat"
sio.savemat(save_path, training_costs);
sess.close()
print("Training events saved in file: %s" % save_path)
