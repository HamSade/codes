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

from data_loader import data_loader, data_parser, pre_emph, de_emph
from conv_gen import auto_encoder

#%% loading data
input_dim =2**14
emphasis = True
n_batch = 2000


#%% Parameters
 
learning_rate = .001 
training_epochs = 1
batch_size      = 128
display_step = 100

dropout_p = 0.5

print('input_dim', input_dim)
print('emphasis', emphasis)
print('learning_rate', learning_rate)
print('batch_size', batch_size)

#%% ##############################################################################

X = tf.placeholder("float", [batch_size, input_dim])
#drop_out_p=tf.placeholder("float", [1])
mode = tf.placeholder("float", None)
        
#%%##############################################################################
# Building the model
if emphasis:
    X= pre_emph(X)
    
y_pred = auto_encoder(X, mode)    

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
        training_data = data_loader(file_idx)
#        n_batch = int(n_training / batch_size)

        for  i in range(n_batch):
            #pre_emph is zero since we do it on X if we want separatrely
            batch_xs = data_parser(training_data, input_dim, batch_size, preemph=0.0, overlap=True) 
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, mode:0.0})
            
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch_", '%02d' % (epoch+1),
                        "File_", '%02d' % (file_idx),
                      "iteration_", '%04d' % (i+1),
                      "cost=", "{:.9f}".format(c))   

print("Optimization Finished!")
#%%##########################################################################
# Training error calculation

training_data = data_loader(9)
training_error = 0
avg_num = 50

for i in range(avg_num):
    sampled_x = data_parser(training_data, input_dim, batch_size)
    training_error += sess.run(cost, feed_dict={X: sampled_x, mode:1.0})

training_error = (training_error/ avg_num) **0.5


#%% Training error calculation

test_data = data_loader(10)
test_error = 0

#y_pred_test = []
#y_true_test = []
y_pred_test = np.zeros([avg_num*batch_size, input_dim])
y_true_test = np.zeros([avg_num*batch_size, input_dim])

for i in range(avg_num):
    sampled_x = data_parser(test_data, input_dim, batch_size)
    y_pred_test_, y_true_test_, test_error_ = sess.run([y_pred, y_true, cost],
                                                feed_dict={X: sampled_x, mode:1.0})
    
    test_error += test_error_
#    y_pred_test += [y_pred_test_]
#    y_true_test += [y_true_test_]
    y_pred_test [ i * batch_size : (i + 1) * batch_size ,:] = y_pred_test_
    y_true_test [ i * batch_size : (i + 1) * batch_size ,:] = y_true_test_
    
test_error = (test_error/ avg_num) ** 0.5

#PRINTING COSTS
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