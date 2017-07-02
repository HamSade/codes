#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:16:33 2017

@author: hsadeghi
"""
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
#from tensorflow.contrib import ffmpeg
import time

#%% file paths

path_clean = "/vol/grid-solar/sgeusers/hsadeghi/segan_data/clean_trainset_wav_16k/"
path_noisy = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/noisy_trainset_wav_16k/'

#%%
def get_file_list(clean=True):
    #FINDING FILE_NAME    
    if clean:
        path = path_clean
        f=open('noisy_list.txt', "r+")        
    else:
        path = path_noisy
        f=open('noisy_list.txt', "r")         
    # finding number of rows (files in teh folder)
    f_list = [] 
    for i in f:
        f_list.append(i)     
    f.close()   
    return f_list, path

#%%
def data_loader(f_list, path, input_dim, batch_size, overlap=False, preemph=0.):
    
#    f_list, path = get_file_list(batch_size)
    num_all_files = len (f_list)
    
    # selecting batch_size number of these names randomly
    file_ind = np.random.randint(0, num_all_files, batch_size)
    
    file_names=[]
    for i in range(batch_size):
        file_names.append( f_list[file_ind[i]].strip('\n') )
        
    # reading data    
    ################################
    data = np.zeros([0])    
    
    for i in range(batch_size):              
        fs, wave = wavfile.read( path + file_names[i] )
        data = np.concatenate((data, wave), axis=0)

    data = data.astype(np.float32)
    data = (2./65535.) * (data - 32767.) + 1.       
        
    n_data=len(data) #.shape[1];

    formatted_data = np.zeros([batch_size, input_dim])
      
    # Selecting batch_size files at random from the list of all files
    for  i in range(batch_size):
        rand_ind = np.random.randint(0, n_data - input_dim * batch_size)
        
        if overlap:
            formatted_data[i,:]= data[ rand_ind + i * int(0.5*input_dim) :\
                          rand_ind + i* int(0.5*input_dim) + input_dim];  # 50% overlap
        else:
            formatted_data[i,:]= data[ rand_ind + i * input_dim :\
                          rand_ind + (i + 1) * input_dim];  # 0 overlap 
    
    if preemph > 0.:
        formatted_data = pre_emph(formatted_data, preemph).astype(np.float32) 
        
    return formatted_data, n_data

#%% test data_loader

start_time = time.time()
f_list, path = get_file_list(True)
print('--- file_list took {} seconds--'.format(time.time()-start_time))


start_time = time.time()
data, n_data= data_loader(f_list, path, 256, 100)
print('--- {} seconds--'.format(time.time()-start_time))


#%%
def pre_emph(x, coeff=0.95):
    x = x.astype(np.float32)
    x0 = np.reshape(x[:,0], [-1,1]) 
    diff = x[:, 1:] - coeff * x[:, :-1]
    concat = np.concatenate((x0, diff), axis=1)
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

