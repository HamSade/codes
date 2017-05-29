#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 21:02:16 2017

@author: hsadeghi
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as si 

from data_loader import pre_emph, de_emph, data_loader
from conv_gen import auto_encoder

#%% loading data
input_dim =512;
emphasis = False

#%% Parameters

learning_rate = .0001 
training_epochs = 5
batch_size      = 64
display_step = 100

dropout_p = 0.5

print('input_dim', input_dim)
print('emphasis', emphasis)
print('learning_rate', learning_rate)
print('batch_size', batch_size)

#%% ##############################################################################

X = tf.placeholder("float", [None, input_dim])
#drop_out_p=tf.placeholder("float", [1])
mode = tf.placeholder("float", None)
        
#%%##############################################################################
# Building the model
if emphasis:
    X= pre_emph(X)
    
y_pred = auto_encoder(X)    

#%% Cost and optimization setup
y_true = X;
cost = tf.reduce_mean( tf.pow( y_true - y_pred , 2))
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)

#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    
#    learning_rate_new = learning_rate_new /2.   
    for file_idx in range(10):       
#        file_name = 'com_concat_signal_{}.mat'.format(file_idx)
        file_name = 'com_concat_signal_{}.mat'.format(file_idx)
        
        training_data, n_training = data_loader(file_name, input_dim, overlap=False)
        n_batch = int(n_training / batch_size)
        for  i in range(n_batch):
            
            start_ind = np.random.randint(0, n_training-batch_size );  # For shuffling
#            batch_xs= training_data[ start_ind* batch_size : (start_ind + 1)* batch_size, :]
            batch_xs= training_data[ start_ind : start_ind + batch_size, :]
    #        batch_xs= tf.reshape(batch_xs, (batch_size,1,input_dim) )       
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, mode:0.0})
            
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch", '%02d' % (epoch+1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i+1),
                      "cost=", "{:.9f}".format(c))   

print("Optimization Finished!")
#%%##########################################################################
# Testing the network performance

training_data, n_training = data_loader('com_concat_signal_9.mat', input_dim, overlap=False)
training_error=sess.run(cost, feed_dict={X: training_data, mode:1.0})**0.5

test_data, n_test = data_loader('com_concat_signal_10.mat', input_dim, overlap=False)

y_pred_test, y_true_test, test_error = sess.run([y_pred, y_true, cost],
                                                feed_dict={X: test_data, mode:1.0})
test_error = test_error ** 0.5

print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#%%##########################################################################
# Savings network
AE_output={};
if emphasis:
    y_pred_test = de_emph(y_pred_test, input_dim)
    y_true_test = de_emph(y_true_test, input_dim)
AE_output['y_pred_test']=y_pred_test
AE_output['y_true_test']=y_true_test

si.savemat("/home/hsadeghi/Dropbox/May/past_codes/conv_AE_output.mat",
           AE_output);
sess.close()