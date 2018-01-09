#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 11:14:23 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
import scipy.io as sio 

from time import time
from spectrogram_loader import data_loader, data_parser


#%% Params
input_dim = 2 ** 13
batch_size = 64
noise_std_test = [0.01]

#%%
start_time = time()

with tf.Session() as sess:
    
    new_saver = tf.train.import_meta_graph('saved_model/model.ckpt.meta')
    new_saver.restore(sess, "saved_model/model.ckpt")
    
    print("Model restored in {:.0f} seconds".format( time() - start_time) )
    
    # Accessing placeholders and operators
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("input:0")
    maxi = graph.get_tensor_by_name("maximum_spec_value:0")
    training = graph.get_tensor_by_name("training_indicator:0")
    noise_std = graph.get_tensor_by_name("noise_std:0")
    coder_output = graph.get_tensor_by_name("coder_output:0")
    bits = graph.get_tensor_by_name("bits_quantizer:0")
#    batch_size = graph.get_tensor_by_name("batch_size:0")
#    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    

    #%% Smaller batch generations
    test_data = data_loader(10, input_dim)    
#    test_error = 0
    avg_num = 50
    y_p = []
    y_t = []
    maximum_spec = np.zeros([avg_num, batch_size])
    quantizer_bits = [[]]
    
    for i in range(avg_num):
        print('test sample {}/{} coded.'.format(i+1, avg_num))
        sampled_x = data_parser(test_data, input_dim, batch_size)  
        
        y_t_, y_p_, maxi_, bits_ = sess.run(
                [X , coder_output, maxi, bits], feed_dict={X: sampled_x[0],
                                                     maxi : sampled_x[1],
                                                     training : 1.,
                                                     noise_std : noise_std_test})        
        if quantizer_bits == [[]]:
            quantizer_bits = list(bits_)
        else:
            quantizer_bits += list(bits_)
#        test_error += test_error_
        if y_t == []:
            y_t = y_t_
            y_p = y_p_
        else:
            y_t = np.concatenate((y_t, y_t_))
            y_p = np.concatenate((y_p, y_p_))
                        
        maximum_spec[i, :] = np.squeeze(maxi_)          
#    test_error = (test_error/ avg_num) ** 0.5
    
    #%% PRINTING COSTS
#    print( 'test_error', "{:.5f}".format(test_error))
    #
    ##%%##########################################################################
    ## Savings network
    AE_output={};    
    AE_output['y_true_test'] = y_t
    AE_output['y_pred_test'] = y_p
        
    AE_output['input_dim'] = input_dim
    AE_output['maximum_spec'] = maximum_spec
    AE_output['quantizer_bits'] = quantizer_bits
    
#    save_path = "/am/roxy/home/hsadeghi/Dropbox/november/coder_output.mat"
    save_path = "coder_output.mat"
    sio.savemat(save_path, AE_output);
    sess.close()
    
    
    
