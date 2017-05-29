#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:20:14 2017

@author: hsadeghi
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import numpy as np

#%% My imports

#from binary_quantizer import binary_quantizer

from data_loader import add_noise, data_loader

from parts_3 import encoder, decoder, discriminator


#%% Original model imports

#from data_loader import read_and_decode, de_emph
#from bnorm import VBN

#from ops import *
#import timeit
#import os

#%%  
""""Parameters"""
#################
display_step = 100
training_epochs = 1
keep_prob_var = 0.5
#self.epoch = args.epoch
#self.d_label_smooth = args.d_label_smooth
#self.z_dim = args.z_dim
#self.z_depth = args.z_depth

init_l1_weight = 100. 
batch_size =  100
g_nl = 'prelu'
preemph = 0.95
epoch = 86
                                                                                    
deconv_type = "deconv"

bias_downconv = True
bias_deconv = True
bias_D_conv = True

# clip D values
#self.d_clip_weights = False
disable_vbn = False

disc_updates = 1

canvas_size = 2**14
deactivated_noise = False

# dilation factors per layer (only in atrous conv G config)
g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# num fmaps for AutoEncoder SEGAN (v1)
g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

d_num_fmaps =  [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

init_noise_std = 0.5
disc_noise_std = tf.Variable(init_noise_std, trainable=False)

#e2e_dataset = args.e2e_dataset

# G's supervised loss weight
l1_weight = 100.
l1_lambda = tf.Variable(l1_weight, trainable=False)
deactivated_l1 = False

d_learning_rate = 0.0002
g_learning_rate = 0.0002

#%%# Segan Model

X = tf.placeholder("float", [batch_size, canvas_size])
noisybatch = add_noise(X)

#mode = tf.placeholder("float", None)  # for quantizing neurons

#%%

#b_q=binary_quantizer(tf)


#%% Reading data and preemphasis

# We do not need this here
#wavbatch = tf.expand_dims(wavbatch, -1)
#noisybatch = tf.expand_dims(noisybatch, -1)

do_prelu=False
if g_nl == 'prelu':
    do_prelu = True

#def encoder(noisy_w, is_ref, z_on=False, do_prelu=False):
#    return h_i, skips   
#def decoder(h_i, skips, do_prelu=False, z=None, z_on=False):
#    return ret_feats == [wave , z]
#def discriminator(wave_in, reuse=True):
#    return d_logit_out 

wavbatch = tf.expand_dims(X, 2)
noisybatch = tf.expand_dims(noisybatch, 2)

try:
    with tf.variable_scope('gen', reuse=True):
        tf.get_variable_scope().reuse_variables()
        h_i, skips = encoder(noisybatch, is_ref=True, scope='enc', do_prelu=True) 
        ref_Gs = decoder(h_i, skips, scope='dec', do_prelu=True)
        print('gen_reuse')
except ValueError:
    with tf.variable_scope('gen'):
        print('gen_creation')
        h_i, skips = encoder(noisybatch, is_ref=True, scope='enc', do_prelu=True) 
        ref_Gs = decoder(h_i, skips, scope='dec', do_prelu=True)
    
ref_G= ref_Gs[0] 
    
D_rl_joint = tf.concat([wavbatch, noisybatch], 2)
print(ref_G.get_shape().as_list())
D_fk_joint = tf.concat([ref_G, noisybatch], 2)


try:
    with tf.variable_scope('disc', reuse=True):
        tf.get_variable_scope().reuse_variables()
        
        d_rl_logits = discriminator(D_rl_joint)
        tf.get_variable_scope().reuse_variables() #Ensures same net for fake and real
        d_fk_logits = discriminator(D_fk_joint)
        print('disc_reuse')
        
except ValueError:
    with tf.variable_scope('disc'):
        print('disc_creation')
        
        d_rl_logits = discriminator(D_rl_joint)
        tf.get_variable_scope().reuse_variables() #Ensures same net for fake and real
        d_fk_logits = discriminator(D_fk_joint)
    

#%% Cost definition

d_rl_loss = tf.reduce_mean(tf.squared_difference(d_rl_logits, 1.))
d_fk_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 0.))
d_loss = d_rl_loss + d_fk_loss

g_adv_loss = tf.reduce_mean(tf.squared_difference(d_fk_logits, 1.))
g_l1_loss = l1_lambda * tf.reduce_mean(tf.abs(tf.add(ref_G, -wavbatch)))
g_loss = g_adv_loss + g_l1_loss


#%% Optimizers
d_opt = tf.train.RMSPropOptimizer(learning_rate = d_learning_rate).minimize(d_loss)
g_opt = tf.train.RMSPropOptimizer(learning_rate = g_learning_rate).minimize(g_loss)
#d_opt = tf.train.AdamOptimizer(config.d_learning_rate,  beta1=beta_1)
#g_opt = tf.train.AdamOptimizer(config.g_learning_rate,  beta1=beta_1)

#%%
#def vbn(self, tensor, name):
#    if self.disable_vbn:
#        class Dummy(object):
#            # Do nothing here, no bnorm
#            def __init__(self, tensor, ignored):
#                self.reference_output=tensor
#            def __call__(self, x):
#                return x
#        VBN_cls = Dummy
#    else:
#        VBN_cls = VBN
#    if not hasattr(self, name):
#        vbn = VBN_cls(tensor, name)
#        setattr(self, name, vbn)
#        return vbn.reference_output
#    vbn = getattr(self, name)
#    return vbn(tensor)

#%% 
""" Training SEGAN """

init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85) #0.333
gpu_options.allow_growth = True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

sess.run(init)

#file_writer = tf.summary.FileWriter('logs/', sess.graph)

#%%

for epoch in range(training_epochs):
    
#    learning_rate_new = learning_rate_new /2.
    
    for file_idx in range(1,11):
        
        file_name = 'com_concat_signal_{}.mat'.format(file_idx)
        
        training_data, n_training = data_loader(file_name, canvas_size)
        
        n_batch = int(2 * n_training / batch_size)
        
        for  i in range(n_batch):
            
            start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
            batch_xs= training_data[ start_ind* batch_size : (start_ind + 1)* batch_size, :]; 
            
            
            for d_iter in range(disc_updates):
                _d_opt, _d_fk_loss, _d_rl_loss =sess.run([d_opt, d_fk_loss, d_rl_loss],feed_dict={X: batch_xs})
            
            # now G iterations
            _g_opt, _g_adv_loss, _g_l1_loss = sess.run([g_opt, g_adv_loss, g_l1_loss],feed_dict={X: batch_xs})  #, mode:0.0})
        
            # Display logs per epoch step
            if i % display_step == 0:
                    
                print("File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i + 1),
                      "d_fk_loss=", "{:.9f}".format(_d_fk_loss),
                      "d_rl_loss=", "{:.9f}".format(_d_rl_loss),
                      "g_adv_loss=", "{:.9f}".format(_g_adv_loss),
                      "g_l1_loss=", "{:.9f}".format(_g_l1_loss))   
                    
print("Optimization Finished!")

#%%##########################################################################
# Testing the network performance
#
#training_data, n_training = data_loader('com_concat_signal_10.mat', canvas_size)
#
#n_batch = int(2 * n_training / batch_size)
#training_d_error  = 0      
#training_g_error  = 0 
#
#for  i in range(n_batch):
#    start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
#    batch_xs= training_data[ start_ind* batch_size : (start_ind + 1)* batch_size, :]; 
#    aa =sess.run([d_loss], feed_dict={X: batch_xs})
#            
#    bb = sess.run([g_loss], feed_dict={X: batch_xs})    
#    
#    training_d_error += aa
#    training_g_error += bb
#
#    
#training_d_error = training_d_error/ n_batch
#training_g_error = training_g_error / n_batch
#    
#print('training_d_error', training_d_error)    
#print('training_g_error', training_g_error)    
#    
#
##############################################    
#test_data, n_test = data_loader('com_concat_signal_11.mat', canvas_size)
#
#n_batch = int(2 * n_test / batch_size)
#test_d_error  = 0      
#test_g_error  = 0 
#
#for  i in range(n_batch):
#    start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
#    batch_xs= training_data[ start_ind* batch_size : (start_ind + 1)* batch_size, :]; 
  
#    aa =sess.run([d_loss], feed_dict={X: batch_xs})
#            
#    bb = sess.run([g_loss], feed_dict={X: batch_xs})    
#    
#    test_d_error += aa
#    test_g_error += bb
#
#    
#test_d_error = training_d_error/ n_batch
#test_g_error = training_g_error / n_batch
#    
#print('test_d_error', training_d_error)    
#print('test_g_error', training_g_error)

#print('learning_rate= ', learning_rate)
#print('num_quantization_steps= ', num_steps)
#%%##########################################################################
# Savings network

#AE_output={};
#AE_output['y_pred_test']=de_emph(y_pred_test, input_dim)
#AE_output['y_true_test']=de_emph(y_true_test, input_dim)
#
#si.savemat("/home/hsadeghi/Dropbox/May/past_codes/rnn_AE_output.mat",
#           AE_output);
#
#sess.close()

