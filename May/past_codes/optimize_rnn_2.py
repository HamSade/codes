#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 23:41:52 2017

@author: hsadeghi
"""
import scipy.io as si 

from rnn_AE_13 import rnn

#%%
class parameters():
    def __init__(self):
        self.input_dim  = [128, 512, 1024]
        self.learning_rate = [ 0.0001]
        self.training_epochs = [10]
        self.full_width = [1024, 512]
        self.batch_size = [32, 128]
        self.rnn_width = [512]
        self.num_steps = [8, 16]
        self.dropout_p=[1.]
        self.emphasis = True
        

#%%

params=parameters();

counter = 1

for i1 in range(len(params.input_dim)):
    for i2 in range(len(params.learning_rate)):
        for i3 in range(len(params.training_epochs)):
            for i4 in range(len(params.full_width)):
                for i5 in range(len(params.batch_size)):
                    for i6 in range(len(params.rnn_width)):
                        for i7 in range(len(params.num_steps)):
                            for i8 in range(len(params.dropout_p)):
                                
                                          
                                try:
                                    a,b = rnn( input_dim        = params.input_dim         [i1],
                                                learning_rate    = params.learning_rate     [i2], 
                                                training_epochs  = params.training_epochs   [i3],
                                                batch_size       = params.full_width        [i4],
                                                full_width       = params.batch_size        [i5],
                                                rnn_width        = params.rnn_width         [i6],
                                                binary_width     = int(params.input_dim[i1]/params.num_steps[i7]*1.45),
                                                num_steps        = params.num_steps         [i7],
                                                dropout_p        = params.dropout_p         [i8],
                                                display_step     = 100, scope=str(counter), emphasis=params.emphasis)
                                    
                                    counter += 1
                                    training_error = a
                                    test_error = b
                                    
                                    
                                    results={};
                                    results['index'            ] = [i1, i2, i3, i4, i5, i6, i7, i8]
                                    results['training_error'    ]= training_error
                                    results['test_error'        ]= test_error
                                    
                                    
                                    results['input_dim']         = params.input_dim         [i1]
                                    results['learning_rate' ]    = params.learning_rate     [i2] 
                                    results['training_epochs' ]  = params.training_epochs   [i3]
                                    results['batch_size' ]       = params.full_width        [i4]
                                    results['full_width'     ]   = params.batch_size        [i5]
                                    results['rnn_width'       ]  = params.rnn_width         [i6]
                                    results['binary_width' ]     = int(params.input_dim[i1]/params.num_steps[i7]*1.45)
                                    results['num_steps'      ]   = params.num_steps         [i7]
                                    results['dropout_p'      ]   = params.dropout_p         [i8]
                                    
                                    
                        
#                                    save_name = "/home/hsadeghi/Dropbox/May/opt_rnn_results/config_cuda_2_{}_{}.mat".format(counter, test_error)
#                                    si.savemat(save_name, results);
#                                   
                                    save_name = "/home/hsadeghi/Dropbox/May/opt_rnn_results/config_cuda_2_{}_{}.txt".format(counter, test_error) 
                                    file_1 = open(save_name, "w") 
                                    for key in results:                                        
                                        file_1.write( key + '   '+ str(results[key])+ '\n')                                    
                                    file_1.close()
                                    
                                    
                                    
                                except ValueError:
                                    print(ValueError)
                                    continue