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

from rectangle_generator import data_generator
#from line_generator import data_generator

from dcgan import generator, discriminator

from time import time
import matplotlib.pyplot as plt

#%% Main parameters

input_dim = 2 ** 5
batch_size = 128
display_step = 30

both_adam = False
apply_clip_weights =    True
clip_limit = 0.9

gpu_frac = 1.0
disc_update_cycles = 1

#n_batch = 1
#training_epochs = 1
#num_training_files = 10

n_batch = 100
training_epochs = 16
num_training_files = 10 

learning_rate_init = 0.0001  #DCGAN = 0.0002
lr_dec_fac = 1.
noise_std_init = [1]
noise_dec_fac = 1.


#%% prINTING

print('noise_std_init', noise_std_init)
print('learning_rate_init', learning_rate_init)
print('input_dim', input_dim)
print('batch_size', batch_size)

#%% ##############################################################################
X = tf.placeholder("float", [batch_size, None, None], name ='spectrogram_input')
training = tf.placeholder("float",None, name = 'training_indicator')
learning_rate = tf.placeholder("float", None, name = 'learning_rate') 
noise_std = tf.placeholder("float", 1, name = 'noise_std')     
maxi = tf.placeholder("float", None, name = 'maximum_spec_value')

#%%##############################################################################
# Building the model
Sxx_real = X
#x_in_gen = Sxx.get_shape().as_list()
x_in_gen = [batch_size, input_dim, input_dim]

with tf.variable_scope('gen'):
    gen_output = generator(x_in_gen, training = training)

with tf.variable_scope('disc'):
    d_real = discriminator( Sxx_real, training = training)
with tf.variable_scope('disc', reuse = True):
    d_fake = discriminator(gen_output, training = training)


#%%
def xentropy(logits, labels, name="x_entropy"):
        y = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
#        tf.scalar_summary(name, xentropy)
        return y
    
    
#%% Cost definitions

# DCGAN
#disc_cost_dcgan =  1/2.* (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_real, labels = tf.ones_like(d_real)))+
#        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.zeros_like(d_fake))) )
#gen_cost_dcgan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_fake, labels = tf.ones_like(d_fake)))

# original GAN
#disc_cost = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
#gen_cost = -tf.reduce_mean(tf.log(d_fake))

##Wasserstein
#d_fake = tf.sigmoid(d_fake)
#d_real = tf.sigmoid(d_real)
disc_cost_w = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
gen_cost_w = -tf.reduce_mean(d_fake)

# xentropy
#disc_cost_real_w = xentropy(d_real, tf.ones_like(d_real))
#disc_cost_fake_w = xentropy(d_fake, tf.zeros_like(d_fake) )
#disc_cost_w = disc_cost_real_w + disc_cost_fake_w
#gen_cost_w = xentropy(d_fake, tf.ones_like(d_fake) )

# Wasserstein
#gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.ones_like(d_real)))
#disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_real, tf.ones_like(d_real))) \
#            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_fake, tf.zeros_like(d_fake)))
    
## LSGAN (mean squared)
#disc_cost_real_mse = tf.reduce_mean( tf.squared_difference(d_real, 1.))
#disc_cost_fake_mse = tf.reduce_mean( tf.squared_difference(d_fake, 0.))
#disc_cost_mse = 0.5 * (disc_cost_real_mse + disc_cost_fake_mse)
#gen_cost_mse = 0.5 * tf.reduce_mean(tf.squared_difference(d_fake, 1.)) 

# Final cost
disc_cost = disc_cost_w
gen_cost = gen_cost_w
#disc_cost = disc_cost_w + disc_cost_mse 
#gen_cost = gen_cost_w + gen_cost_mse
#disc_cost = disc_cost_mse 
#gen_cost = gen_cost_mse
#disc_cost = disc_cost_dcgan
#gen_cost = gen_cost_dcgan


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
#    print('disc_variable', x.name)
    assert x not in gen_vars
for x in gen_vars:
#    print('gen_variable', x.name)
    assert x not in disc_vars
for x in t_vars:
#    print('all_variables', x.name)
    assert x in gen_vars or x in disc_vars, x.name
if apply_clip_weights:
#    print('Clipping D weights')
    clip_disc_vars = [v.assign(tf.clip_by_value(v, -1 * clip_limit, clip_limit)) for v in disc_vars]
else:
    print('Not clipping D weights')
    
#%% Optimizers
if both_adam:
    disc_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(disc_cost, var_list=disc_vars)
    gen_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(gen_cost, var_list=gen_vars)
    total_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(gen_cost + disc_cost)
else:
    disc_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(disc_cost, var_list = disc_vars)
    gen_opt  = tf.train.RMSPropOptimizer(learning_rate).minimize(gen_cost , var_list = gen_vars )
    total_opt = tf.train.RMSPropOptimizer(learning_rate).minimize(gen_cost + disc_cost)
    
    
#%%##############################################################################
# Training
#init = tf.global_variables_initializer()
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_frac)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement = False, gpu_options=gpu_options))
#sess.run(init)

init = tf.global_variables_initializer()
myconfig = tf.ConfigProto()   
myconfig.log_device_placement = False
myconfig.gpu_options.allow_growth=True    
sess = tf.Session(config=myconfig)
sess.run(init)

#%% Tensorboard setup
#tf.summary.scalar('gen_l2_cost', gen_l2_cost)
#tf.summary.scalar('gen_adv_cost', gen_adv_cost)
#tf.summary.scalar('disc_cost', disc_cost)
#merged_summary = tf.summary.merge_all()
#summary_writer = tf.summary.FileWriter('/home/hsadeghi/Dropbox/august/spectrogram/tensorboard/dc_BWE/1/')
#summary_writer.add_graph(sess.graph)
 
#%% Training cycle
start_time =  time()
counter = 0
for epoch in range(training_epochs):
    for file_idx in range(num_training_files): 
        for  i in range(n_batch):
            
            counter += 1
            
            ################################################################
            batch_xs = data_generator(input_dim, batch_size)  
            # Update disc
#            with tf.device('/gpu:0'):
            for _ in range(disc_update_cycles):
                _, disc_cost_ = sess.run([disc_opt, disc_cost],
                                         feed_dict={X: batch_xs, maxi : 1.,
                                                    learning_rate: learning_rate_init,
                                                    noise_std : noise_std_init,
                                                    training : 1.})

            # clip disc weights
            if apply_clip_weights:
                sess.run(clip_disc_vars)
            
            ####################################1############################
            # Update generator
#            with tf.device('/gpu:1'):
            batch_xs = data_generator(input_dim, batch_size)  # Hack #4
            _, gen_cost_ = sess.run([gen_opt, gen_cost], feed_dict={X: batch_xs,
                                              maxi : 1.,
                                              learning_rate: learning_rate_init,
                                              noise_std : noise_std_init,
                                              training : 1.})
#            if gen_cost_ < 5010:
#                learning_rate_init = learning_rate_init/10
#           Display logs per epoch step

            
#                summary_writer.add_summary(ss, i)  #tensorboard

            time_lapsed = time()-start_time
            print("epoch:", '%02d' % (epoch + 1),
                    "File:", '%02d' % (file_idx),
                  "iteration:", '%04d' % (i + 1),
                  "disc_cost =", "{:.9f}".format( (10**4) * disc_cost_),
                  "gen_adv_cost =", "{:.9f}".format( (10**4) * gen_cost_),
                  'time lapsed =', '%02d' % (time_lapsed//3600), 'h:', '%02d'%((time_lapsed//60)%60),'m')
            
            if counter % (display_step) == 0:
                plt.figure(1)
                plt.plot(counter, (10**4) * disc_cost_, 'r*') 
                plt.hold(True)
                plt.pause(0.01)
                  
                plt.plot(counter, (10**4) * gen_cost_, 'b.') 
                plt.hold(True)
                plt.pause(0.01)
                
                plt.legend('Disc_cost','Gen_cost')
                
                #########display some sample spectrogram results
#                if i % display_step * 100 == 0:
                plt.figure(2)
                gen_output_ = sess.run([gen_output], feed_dict={X: batch_xs,
                                              maxi : 1., training : 0.})
                plt.pcolormesh(gen_output_[0][8])
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.pause(0.01)
                
            #### Updating both once in a while
            ################################################################
            if counter % (10*counter) == 0:
                batch_xs = data_generator(input_dim, batch_size)  # Hack #4
                _ = sess.run([total_opt], feed_dict={X: batch_xs,
                                              maxi : 1.,
                                              learning_rate: learning_rate_init,
                                              noise_std : noise_std_init,
                                              training : 1.})
            
            
            
    
        learning_rate_init = max( 0.0001, learning_rate_init /lr_dec_fac)
        noise_std_init[0] =   max(0.01, noise_std_init[0]/ noise_dec_fac  )
    reading_phase = False
    
#    plt.savefig(1, '/home/hsadeghi/Desktop/costs.png')
#    plt.savefig(2, '/home/hsadeghi/Desktop/last_sample.png')
print("Optimization Finished!")

#%%##########################################################################
# Training error calculation
training_error = 0
avg_num = 50
for i in range(avg_num):
    sampled_x = data_generator(input_dim, batch_size)
    training_error += sess.run(gen_cost, feed_dict={X: sampled_x,
                                              maxi : 1., noise_std : [0.],
                                                    training : 0. })
training_error = (training_error/ avg_num) ** 0.5

#%% Test error calculation
test_error = 0
avg_num = 50

#n_time_bins = 64
#nfft = 128

y_p = [] #np.zeros([ avg_num*(batch_size), nfft, n_time_bins])
y_t = [] # np.zeros([ avg_num*(batch_size), nfft, n_time_bins])   
maximum_spec = np.zeros(avg_num)

for i in range(avg_num):
    sampled_x = data_generator(input_dim, batch_size) 
    y_true_test_, y_pred_test_,\
    test_error_, maxi_ = sess.run([Sxx_real , gen_output, gen_cost, maxi],\
                                        feed_dict={X: sampled_x,
                                              maxi : 1.,
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

save_path = "/home/hsadeghi/Dropbox/september/rectangle_generation/dcgan_spectrogram_output.mat"
sio.savemat(save_path, AE_output);
sess.close()