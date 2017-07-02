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
import scipy.io as sio 
from data_loader import data_loader, data_parser, data_parser_2, pre_emph, de_emph, apply_imdct
from conv_gen import auto_encoder

#%% Main parameters

input_dim =2**9
overlap= input_dim #int(3*input_dim/2)

emphasis = False
mdct_indicator= False
apply_mask = True #Whther to cut the test input

#n_batch = 1
#training_epochs = 1
#num_training_files = 1

n_batch = 3000
training_epochs = 3
num_training_files = 10

learning_rate = 0.0001


#%% Other Parameters

batch_size = 128
display_step = 100

dropout_p = 1.

print('emphasis', emphasis)
print('mdct_indicator', mdct_indicator)
print('overlap', overlap)
print('input_dim', input_dim)
print('emphasis', emphasis)
print('learning_rate', learning_rate)
print('batch_size', batch_size)

#%% Rectanular mask buildup 
zeros = np.zeros(int(overlap))
flat = np.ones(input_dim)
mask = np.concatenate([zeros, flat, zeros])
        
#%% Trapezoid mask buildup 
#zeros = np.zeros(int(overlap/2))
#ramp_1 = np.array(range(int(overlap/2)))/(overlap/2. - 1)
#flat = np.ones(input_dim)
#ramp_2 = 1-ramp_1
#mask = np.concatenate([zeros, ramp_1, flat, ramp_2, zeros])
#        
#%% ##############################################################################

X_ = tf.placeholder("float", [batch_size, input_dim + overlap * 2])
#drop_out_p=tf.placeholder("float", [1])
mode = tf.placeholder("float", None)
        
#%%##############################################################################
# Building the model
if emphasis:
    X= pre_emph(X_)
else:
    X=X_

y_true = X;    
y_pred = auto_encoder(X, mode) 

#%% Cost and optimization setup
cost = tf.reduce_mean( tf.pow( y_true - y_pred , 2))
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)

if emphasis:
    y_pred= de_emph(y_pred)
    y_true = de_emph(y_true)
    
#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#%% Checking variables
#tvars = tf.trainable_variables()
#for v in tvars:
#    print(v.name)

#%% Training cycle
for epoch in range(training_epochs):
    for file_idx in range(num_training_files):

        if mdct_indicator:               
            training_data, _, _ = data_loader(file_idx, input_dim, mdct_indicator)
        else:
            training_data= data_loader(file_idx, input_dim, mdct_indicator)
        for  i in range(n_batch):
            #pre_emph is zero since we do it on X if we want separatrely
            
            if mdct_indicator:
                batch_xs = data_parser_2(training_data, batch_size, overlap = overlap) 
            else:
                batch_xs = data_parser(training_data, input_dim, batch_size, overlap=overlap) 
            
            _, c = sess.run([optimizer, cost], feed_dict={X_: batch_xs, mode:0.0})
            
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch:", '%02d' % (epoch+1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i+1),
                      "cost =", "{:.9f}".format( (10**4) *c)) 
            
        learning_rate = learning_rate /np.sqrt(2)                
    learning_rate = learning_rate /np.sqrt(2)                 
        
print("Optimization Finished!")
#%%##########################################################################
# Training error calculation

if mdct_indicator:               
    training_data, _, _ = data_loader(9, input_dim, mdct_indicator)
else:
    training_data= data_loader(9, input_dim, mdct_indicator)
training_error = 0
avg_num = 100

for i in range(avg_num):
    if mdct_indicator:
        sampled_x = data_parser_2(training_data, batch_size, overlap=overlap, apply_mask=apply_mask)
    else:
        sampled_x = data_parser(training_data, input_dim, batch_size, overlap=overlap, apply_mask=apply_mask)
    training_error += sess.run(cost, feed_dict={X_: sampled_x, mode:1.0})

training_error = (training_error/ avg_num) ** 0.5


#%% Test error calculation
if mdct_indicator:               
    test_data, max_value, mean_value = data_loader(10, input_dim, mdct_indicator)
else:
    test_data = data_loader(10, input_dim, mdct_indicator)    
test_error = 0
avg_num = 50

#y_pred_test = []
#y_true_test = []

if mdct_indicator:
    y_pred_test = np.zeros([avg_num*(batch_size-1), input_dim])
    y_true_test = np.zeros([avg_num*(batch_size-1), input_dim])
else:
    y_pred_test = np.zeros([avg_num*(batch_size), input_dim + overlap *2])
    y_true_test = np.zeros([avg_num*(batch_size), input_dim + overlap *2])
    
for i in range(avg_num):
    if mdct_indicator:
        sampled_x = data_parser_2(test_data, batch_size, overlap=overlap, apply_mask=apply_mask)
    else:
        sampled_x = data_parser(test_data, input_dim, batch_size, overlap=overlap, apply_mask=apply_mask)
    
    y_pred_test_, y_true_test_, test_error_ = sess.run([y_pred, y_true, cost],
                                                feed_dict={X_: sampled_x, mode:1.0})
    
    test_error += test_error_
        
    if mdct_indicator:
        y_pred_test_ = np.transpose(y_pred_test_ * max_value)
        y_true_test_ = np.transpose(y_true_test_ * max_value)
        y_pred_test_ = apply_imdct( y_pred_test_, input_dim )
        y_true_test_ = apply_imdct( y_true_test_, input_dim )  
        y_pred_test_ = np.reshape(y_pred_test_, [batch_size-1, input_dim])
        y_true_test_ = np.reshape(y_true_test_, [batch_size-1, input_dim])  
        y_pred_test [ i * batch_size : (i + 1) * batch_size -1 ,:] = y_pred_test_
        y_true_test [ i * batch_size : (i + 1) * batch_size -1,:] = y_true_test_
        
    else:
        y_pred_test [ i * batch_size : (i + 1) * batch_size  ,:] = y_pred_test_
        y_true_test [ i * batch_size : (i + 1) * batch_size ,:] = y_true_test_
        
test_error = (test_error/ avg_num) ** 0.5

#PRINTING COSTS
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))

#%%##########################################################################
# Savings network
AE_output={};
AE_output['y_true_test'] = y_true_test
AE_output['y_pred_test'] = y_pred_test
AE_output['input_dim'] = input_dim
AE_output['overlap'] = overlap
save_path = "/home/hsadeghi/Dropbox/june/conv_codec/conv_AE_output.mat"
sio.savemat(save_path, AE_output);
sess.close()