#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:10:00 2017
@author: hsadeghi
"""
#import tensorflow as tf
import numpy as np
import scipy.io as si 
import scipy.signal as ss 
import matplotlib.pyplot as plt
#import librosa as libr

#%%
#path_name = '/vol/grid-solar/sgeusers/hsadeghi/data/'
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/'
#path_name = '/vol/grid-solar/sgeusers/hsadeghi/simulated_data/poisson_pulse_train/'

fs = 16000
fc = 4000
wn = fc/ (fs/2)  #cutoff for speech    

#%%
def data_loader(file_ind, input_dim): 
#    file_name = path_name + 'com_concat_signal_{}.mat'.format(file_ind) 
    file_name = path_name + 'clean_{}.mat'.format(file_ind)
#    file_name = path_name + 'data_{}.mat'.format(file_ind)

    mat = si.loadmat(file_name)  
    data = mat['concat_wav']
    data = np.array(data) #data.shape = (1, 47456861)
    # clip length to be a multiple of 2048
    n_data = data.shape[1];
    n_data = n_data - n_data % input_dim
    data = data[0,0:n_data] # data.shape = (47456768,)
    return data
    
#%%
def data_parser(data, input_dim, batch_size):    
    data = np.reshape(data, [1,-1])
    n_data=data.shape[1];
    parsed_data = np.zeros([batch_size, input_dim])
    rand_ind = np.random.randint( 0 , n_data - batch_size * input_dim )

    for  i in range(batch_size):  # +1 is for the parser to produce n_batch spectrograms 
        data_window = data[0, rand_ind + i * (input_dim) :\
                                  rand_ind + (i+1) * (input_dim) ]  
        parsed_data[i, 0:input_dim] = data_window
        
    return band_split(parsed_data)
 
#%%
def band_split(x): # input_dim is : batch_size, input_dim + overlap * 2 
    
    # make x 1D
    n_batch = x.shape[0]
    # input_dim = x.shape[-1]
    x = np.reshape(x, [-1])
    
    nperseg = int( 16/1000 * 16000 )  #input_dim // 8 #
    noverlap = nperseg // 2
#    nfft = input_dim
    f, t, Sxx = ss.spectrogram(x, fs = fs,
                               window = ('tukey', 0.25), nperseg = nperseg,
                               noverlap = noverlap, # nfft = nfft,
                               return_onesided = True, axis = -1, mode = 'magnitude')

    Sxx = 20. *  np.log10(Sxx) 
    # Smoothing noisy spectrogram
    # Sxx = ss.convolve2d(Sxx, np.ones([10,10]), mode='same', boundary='wrap')
    
    #### normalizing to [-1, 1]
    Sxx = Sxx - Sxx.mean(axis=-1).reshape(-1, 1)
    maxi =  np.max(np.abs(Sxx))
    Sxx = Sxx / maxi    
    
    # Concatenating zeros
    Sxx =  np.concatenate((Sxx, np.zeros(shape=[Sxx.shape[0],1])), axis = -1)
#    removing one extra freq bin
    Sxx = Sxx [:-1, :]
    
    Sxx = np.asarray(np.array_split(Sxx, n_batch, axis = -1))
    return Sxx, maxi

#%%
#x = np.random.normal(size=[128 , 2**11]) 
#Sxx_l, Sxx_h, maxi  =  band_split(x)

input_dim = 2 ** 16
n_batch = 1
x = data_loader(1, input_dim)
Sxx, maxi = data_parser(x, input_dim, n_batch)

print('Sxx.shape after split', [len(Sxx), Sxx[0].shape])

rand_ind = np.random.randint(low = 0, high = Sxx.shape[0])
plt.pcolormesh(Sxx[rand_ind,:,:])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#print('std of Spectrogram', np.mean(np.sqrt(np.mean(np.power(Sxx_h - np.mean(Sxx_h,axis=0), 2), axis=0))))
