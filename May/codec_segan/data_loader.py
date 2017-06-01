#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:01:33 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
import scipy.io as si 
import time


#%%
#path_name = '/vol/grid-solar/sgeusers/hsadeghi/data/'
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/'


#%%
def data_loader(file_ind):
    file_name = path_name + 'clean_{}.mat'.format(file_ind) 
#    file_name = path_name + 'com_concat_signal_{}.mat'.format(file_ind) 
    
    mat = si.loadmat(file_name)  
    #    fs=np.asscalar(mat['fs_new']);
    data=mat['concat_wav']
    data=np.array(data)
    return data

#%%
def data_parser(data, input_dim, batch_size, preemph=0.0, overlap=False):
    
    n_data=data.shape[1];

    parsed_data = np.zeros([batch_size, input_dim])
    
    if overlap:
        rand_ind = np.random.randint(0, n_data - int(2/3.* input_dim * batch_size) - 1)
        for  i in range(batch_size):    
            parsed_data[i,:]= data[0, rand_ind + i * int(0.5*input_dim) :\
                          rand_ind + i* int(0.5*input_dim) + input_dim];  # 50% overlap
    else:
        rand_ind = np.random.randint(0, n_data - input_dim * batch_size)
        ### Slow parsing
#        for  i in range(batch_size): 
#             parsed_data[i,:]= data[0, rand_ind + i * input_dim :\
#                       rand_ind + (i + 1) * input_dim];  # 0 overlap 
        ### Fast parsing                
        parsed_data = np.reshape( data [0, rand_ind: rand_ind + batch_size * input_dim],
                                 [batch_size, input_dim])
        
        
#    parsed_data = pre_emph(parsed_data, preemph)    

    return parsed_data

#%%
def pre_emph(x, coeff=0.95):
    x = tf.cast(x, tf.float32)
#    x = x.astype(np.float32)
    x0 = tf.reshape(x[:,0], [-1,1]) 
    diff = x[:, 1:] - coeff * x[:, :-1]
    concat = tf.concat((x0, diff), 1)
    return concat


#%%*
def de_emph(y, len_y, coeff=0.95):
    y = y.astype(np.float32)
    if coeff <= 0:
        return y
    x = y[:, 0]
    x= np.reshape(x , [-1,1])    
    for n in range(1, len_y):
        new_col = np.add(coeff * x[:, n - 1] , y[:,n])
        new_col = np.reshape(new_col , [-1,1])
        x = np.concatenate([x, new_col],1)
    return x



#%% testing de_emph

#start_time = time.time()
#data = data_loader(1)
#print('---data loading = {} seconds--'.format(time.time()-start_time))
#
#start_time = time.time()
#data = data_parser(data, 2**14, 128)
#print('---data parsing = {} seconds--'.format(time.time()-start_time))
