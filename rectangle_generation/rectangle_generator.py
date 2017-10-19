#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 13:27:39 2017

@author: hsadeghi
"""

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
        sxx_temp = np.zeros(shape = [n_row, n_col])
        
        # Random side lenth
        side_row = np.random.randint(low = 1, high = n_row//4)
        side_col = np.random.randint(low = 1, high = n_col//4)
        side_length = min(side_row, side_col)
        # det side length
#        side_length = 20
        
        
        cen_row = np.random.randint(low = side_length, high = n_row - side_length)
        cen_col = np.random.randint(low = side_length, high = n_col - side_length)
        
        sxx_temp [cen_row - side_length: cen_row + side_length,
                  cen_col - side_length: cen_col + side_length] = 0.9
                  
#        side_row = np.random.randint(low = 1, high = n_row//4)
#        side_col = np.random.randint(low = 1, high = n_col//4)
#        cen_row = np.random.randint(low = side_row, high = n_row - side_row)
#        cen_col = np.random.randint(low = side_col, high = n_col - side_col)
#        sxx_temp [cen_row - side_row: cen_row + side_row,
#                  cen_col - side_col: cen_col + side_col] = 1 
        Sxx.append(sxx_temp)
        
    Sxx = np.asarray(Sxx)
    return Sxx


#%% test
#input_dim = 2 ** 4
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
