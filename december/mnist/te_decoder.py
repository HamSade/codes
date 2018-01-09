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
from mnist_loader import data_loader, data_parser


#%% Params
input_dim = 2 ** 13
batch_size = 128
down_ratio = 4
noise_std_test = [1.0]

#%%
start_time = time()

with tf.Session() as sess:
    
    load_path = "/vol/grid-solar/sgeusers/hsadeghi/research_results/"
    
    new_saver = tf.train.import_meta_graph(load_path + 'saved_model/model.ckpt.meta')
    new_saver.restore(sess, load_path + "saved_model/model.ckpt")
    
    print("Model restored in {:.0f} seconds".format( time() - start_time) )
    
    # Accessing placeholders and operators
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("input:0")
    maxi = graph.get_tensor_by_name("maximum_spec_value:0")
    training = graph.get_tensor_by_name("training_indicator:0")
    noise_std = graph.get_tensor_by_name("noise_std:0")
    coder_output = graph.get_tensor_by_name("coder_output:0") 

    #%% Smaller batch generations
    test_data = data_loader()    
#    test_error = 0
    avg_num = 50
    y_p = []
    y_t = []
    maximum_spec = np.zeros([avg_num, batch_size])
    quantizer_bits = [[]]
    
    for i in range(avg_num):
        print('test sample {}/{} coded.'.format(i+1, avg_num))
        sampled_x = data_parser(test_data, batch_size)  
        
        y_t_, y_p_, maxi_ = sess.run(
                [X , coder_output, maxi], feed_dict={X: sampled_x[0],
                                                     maxi : sampled_x[1],
                                                     training : 1.,
                                                     noise_std : noise_std_test})        
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
    
#    save_path = "coder_output.mat"
    save_path = "/vol/grid-solar/sgeusers/hsadeghi/research_results/saved_model/coder_output.mat"
    sio.savemat(save_path, AE_output);
    sess.close()
    
    
    
