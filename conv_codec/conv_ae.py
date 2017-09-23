#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 23:29:10 2017

@author: hsadeghi

2-band--- all lowpass
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.io as sio 
from data_loader import data_loader, data_parser, data_parser_2
from data_loader import pre_emph, de_emph, apply_imdct, band_merge
from conv_gen_2 import auto_encoder

#%% Main parameters

input_dim =2**9
overlap= int(input_dim/2**7)

emphasis = False
mdct_indicator= False
apply_mask = False #Whether to cut the test input

comp_l = 2
comp_h = 2

lambda_h = 10**6

n_batch = 1
training_epochs = 1
num_training_files = 1

n_batch = 2000
training_epochs = 10
num_training_files = 3

learning_rate_init = 0.01


#%% Other Parameters

batch_size = 128
display_step = 100

dropout_p = 1.

print('lambda_h', lambda_h)
print('comp_l', comp_l)
print('comp_h', comp_h)
print('emphasis', emphasis)
print('mdct_indicator', mdct_indicator)
print('overlap', overlap)
print('input_dim', input_dim)
print('emphasis', emphasis)
print('learning_rate_init', learning_rate_init)
print('batch_size', batch_size)

#%% Rectanular mask buildup 
zeros = np.zeros(int(overlap))
flat = np.ones(input_dim)
mask = np.concatenate([zeros, flat, zeros])
      
#%% ##############################################################################

X = tf.placeholder("float", [2, batch_size, input_dim + overlap * 2])
#drop_out_p=tf.placeholder("float", [1])
mode = tf.placeholder("float", None)
learning_rate = tf.placeholder("float", None) 
      
#%%##############################################################################
# Building the model
X_l = X[0]
X_h = X[1]
if emphasis:
    X_h = pre_emph(X_h)

codes=[]
with tf. variable_scope('low'):
    y_pred_l, codes_temp = auto_encoder(X_l, mode, comp_l)
    codes.append(codes_temp)
with tf. variable_scope('high'):
    y_pred_h, codes_temp = auto_encoder(X_h, mode,comp_h)
    codes.append(codes_temp)

# Type casting
codes =  tf.concat(codes, axis=-1)

y_true_l = X_l
y_true_h = X_h

#%% Cost and optimization setup
cost_l = tf.reduce_mean( tf.pow( tf.add(y_true_l, -y_pred_l) , 2))
cost_h = tf.reduce_mean( tf.pow( tf.add(y_true_h, -y_pred_h) , 2))
cost = tf.add( cost_l, lambda_h * cost_h)

if emphasis:
    y_true_h = de_emph(y_true_h)
    y_pred_h = de_emph(y_pred_h)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)
#%%##############################################################################
# Training
init = tf.global_variables_initializer()
sess=tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
            
#            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, mode:0.0})
            _, c , cl, ch = sess.run([optimizer, cost, cost_l, cost_h],
                                     feed_dict={X: batch_xs, mode:0.0,
                                                learning_rate: learning_rate_init})
        # Display logs per epoch step
            if i % display_step == 0:
                print("epoch:", '%02d' % (epoch+1),
                        "File:", '%02d' % (file_idx),
                      "iteration:", '%04d' % (i+1),
                      "cost_l =", "{:.9f}".format( (10**4) *cl),
                      "cost_h =", "{:.9f}".format( (10**4) *ch)) 
            
        learning_rate_init = learning_rate_init /1.1 #np.sqrt(2)                
#    learning_rate_init = learning_rate_init / np.sqrt(2)                 
        
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
    training_error += sess.run(cost, feed_dict={X: sampled_x, mode:1.0})

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
    
    y_true_l_, y_true_h_, y_pred_l_, y_pred_h_,\
    test_error_, codes_ = sess.run([y_true_l, y_true_h, \
                                      y_pred_l, y_pred_h, cost, codes],\
                                        feed_dict={X: sampled_x, mode:1.0})

    y_true_test_ = band_merge(y_true_l_, y_true_h_)
    y_pred_test_ = band_merge(y_pred_l_, y_pred_h_)
        
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
AE_output['codes'] = codes_
AE_output['overlap'] = overlap

save_path = "/home/hsadeghi/Dropbox/july/conv_codec/conv_AE_output.mat"
sio.savemat(save_path, AE_output);
sess.close()