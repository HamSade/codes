#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 11:24:43 2017

@author: hsadeghi
"""

#%%

#import tensorflow as tf
import numpy as np
#import scipy.io as si 
#import scipy.signal as ss 
import matplotlib.pyplot as plt

#%%
def data_generator(input_dim, batch_size): 
    n_row = input_dim
    n_col = input_dim
    
    Sxx = [] 
    
    for i in range(batch_size):
        sxx_temp = np.zeros(shape = [n_row, n_col]) + 0.001
        
        num_lines =  np.random.randint(low = 1, high = 10)
#        num_lines = 1
        
        
        row_ind  = np.random.randint(low = 0, high = n_row, size = [num_lines])
#        row_ind = [n_row // 2]
        
        for j in range(num_lines):
            sxx_temp [row_ind[j], :] = 0.9
                  
        Sxx.append(sxx_temp)
        
    Sxx = np.asarray(Sxx)
    return Sxx


#%% test
#input_dim = 2 ** 5
#n_batch = 8
#
#Sxx= data_generator ( input_dim, n_batch)
#
#print('Sxx.shape after split', [len(Sxx), Sxx[0].shape])
#
#rand_ind = np.random.randint(low = 0, high = Sxx.shape[0])
#plt.pcolormesh(Sxx[rand_ind,:,:])
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
