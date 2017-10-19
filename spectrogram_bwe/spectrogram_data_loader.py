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



plot = 0


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

#    Sxx = band_split(parsed_data)
#    return Sxx 
#%%
def band_split(x): # input_dim is : batch_size, input_dim + overlap * 2 
    
    #make x 1D
    n_batch = x.shape[0]
#    input_dim = x.shape[-1]
    x = np.reshape(x, [-1])
    
    nperseg = int( 20 /1000 * 16000 )  #input_dim // 8 #
    noverlap = nperseg * 3 // 4
#    nfft = input_dim
    f, t, Sxx = ss.spectrogram(x, fs = fs,
                               window=('tukey', 0.25), nperseg = nperseg,
                               noverlap = noverlap, # nfft = nfft,
                               return_onesided=True, axis=-1, mode='magnitude')

    Sxx = 20. * np.log10(Sxx) 
    
    # Smoothing noisy spectrogram
#    Sxx = ss.convolve2d(Sxx, np.ones([10,10]), mode='same', boundary='wrap')
    
    #### normalizing to [-1, 1]
    Sxx = Sxx - Sxx.mean(axis=-1).reshape(-1, 1)
    maxi =  np.max(np.abs(Sxx))
    Sxx = Sxx / maxi * 0.9

#    maxi =  np.mean(np.abs(Sxx))    
#    print('Sxx.shape', Sxx.shape)
    
    n_freq_bins = Sxx.shape[0]
    Sxx_l = Sxx[0 : n_freq_bins // 2 + 1, :]
    Sxx_h = Sxx[n_freq_bins // 2 + 1 :n_freq_bins, :] 
    
    
#    print('Sxx_l shape', Sxx_l.shape)
#    print('Sxx_h shape', Sxx_h.shape)
#    

    # Concatenating zeros
    num_cols = Sxx_l.shape[-1]
    n = num_cols //  n_batch + 1 
    num_added_cols = n * n_batch - num_cols
    Sxx_l =  np.concatenate((Sxx_l, np.zeros(shape=[Sxx_l.shape[0],num_added_cols])), axis = -1)
    Sxx_h =  np.concatenate((Sxx_h, np.zeros(shape=[Sxx_h.shape[0],num_added_cols])), axis = -1)
    
    
#    print('Sxx_l shape', Sxx_l.shape)
#    print('Sxx_h shape', Sxx_h.shape)
    

#    print('Sxx_l.shape before split', Sxx_l.shape) #129 x 512
#    print('Sxx_h.shape before split', Sxx_h.shape)

    Sxx_l = np.asarray(np.array_split(Sxx_l, n_batch, axis = -1))
    Sxx_h = np.asarray(np.array_split(Sxx_h, n_batch, axis = -1))
#    
    return Sxx_l, Sxx_h, maxi
#    return Sxx

#%%

if plot==1:
    
    input_dim = 2 ** 13
    x = data_loader(1, input_dim)
    Sxx_l, Sxx_h, maxi = data_parser(x, input_dim, 128)
    #
    #x = np.random.normal(size=[128 , 2**11]) 
    #Sxx_l, Sxx_h, maxi  =  band_split(x)
    #
    #print('std of Spectrogram', np.mean(np.sqrt(np.mean(np.power(Sxx_h - np.mean(Sxx_h,axis=0), 2), axis=0))))
    ##
    print('Sxx_l.shape after split', [len(Sxx_l), Sxx_l[0].shape])
    print('Sxx_h.shape after split', [len(Sxx_h), Sxx_h[0].shape])
    ##
    #print('Sxx_h.max', Sxx_h.max())
    #print('Sxx_h.min', Sxx_h.min())
    #
    plt.pcolormesh(Sxx_l[0])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.pause(1)

#%% Spectrogram sanity check
#x =  data_loader(0, 512);
#Sxx = data_parser(x, 512, 128)
#
#print('Sxx.max', Sxx.max())
#print('Sxx.min', Sxx.min())
#
#print('Sxx.size', Sxx.shape)
#
#for i in range(10):
#    
#    rand_ind =  np.random.randint(0, high = Sxx.shape[1], size=[1])
#    
#    col = Sxx[:, rand_ind]
#    
#    plt.figure()
#    plt.plot(col)
#    plt.savefig('{}.png'.format(i))

#%%
    # USING LIBROSA
#    x_stft = libr.stft(x, n_fft = input_dim, hop_length=None, win_length=None, window='hann',
#         center=True, pad_mode='reflect')
#    Sxx = 20. * np.log10(np.abs(x_stft))
