#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:01:33 2017

@author: hsadeghi
"""
#import tensorflow as tf
import numpy as np
import scipy.ndimage.filters as snf
import scipy.io as si 
#import scipy.signal as ss 
import matplotlib.pyplot as plt

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
    #    fs=np.asscalar(mat['fs_new']);
    data = mat['concat_wav']
    data = np.array(data) #data.shape = (1, 47456861)
    # clip length to be a multiple of 2048
    n_data = data.shape[1];
    n_data = n_data - n_data % input_dim
    data = data[0,0:n_data] # data.shape = (47456768,)
    
    # Downsampling to 8 KHz
#    data = data.astype(np.float64)
#    data = ss.decimate(data, 2)
    return data

#%%
def data_parser(data, input_dim, batch_size, overlap=0, apply_mask=False):    
    data = np.reshape(data, [1,-1])
    n_data=data.shape[1];
    parsed_data = np.zeros([batch_size, input_dim + overlap * 2 ])
    rand_ind = np.random.randint( 0 , n_data - batch_size * (input_dim + overlap * 2))
    # Rect mask
    zeros = np.zeros(int(overlap))
    flat = np.ones(input_dim)
    mask = np.concatenate([zeros, flat, zeros])

    if apply_mask:
        for  i in range(batch_size): 
            data_window = data[0, rand_ind + i * (input_dim) :\
                                      rand_ind + (i+1) * (input_dim) + overlap*2 ]  
            parsed_data[i, 0:input_dim + overlap * 2] = np.multiply(mask, data_window)        
    else:       
        for  i in range(batch_size): 
            data_window = data[0, rand_ind + i * (input_dim) :\
                                      rand_ind + (i+1) * (input_dim) + overlap*2 ]  
            parsed_data[i, 0:input_dim + overlap * 2] = data_window
    
    return band_split(parsed_data)
    
#%%
def band_split(x): # input_dim is : batch_size, input_dim + overlap * 2
    x_f = np.fft.fft(x, axis=-1)   
    length = x.shape[-1]   
    nc = int(wn * (length // 2))  #index of cutoff
    mask_l = np.ones([length])
    mask_l [ nc : length- nc + 1 ] = 0  # +1 comes from the middle frequency fft(length/2 (zero indexing))
    mask_l.astype(np.complex128)
    lpf = np.multiply(x_f, mask_l);
    hpf = np.subtract(x_f, lpf);  

    hpf_l = np.zeros([hpf.shape[0], length], dtype=np.complex128)
    hpf_l [:, 0: length//2 - nc + 1 ] = hpf[ :, nc : length//2 + 1]
    hpf_l [:, length//2 + nc: length]    = hpf[ :, length//2: length - nc ]

    #removing half of samples
    lpf = lpf[:, 0 : length//4]
    hpf_l = hpf_l[:, 0 : length//4]
    
    # calculating dB value of amplitude
    lp = np.abs(lpf)
    hp = np.abs(hpf_l)
#    lp = 20 * np.log10( np.abs(lpf))
#    hp = 20 * np.log10(np.abs(hpf_l))
    
    # smoothening
#    lp = snf.gaussian_filter1d(lp, sigma=15, axis=-1)
    hp = snf.gaussian_filter1d(hp, sigma=10, axis=-1)
    
    # Normalization
#    lp = lp - lp.mean(axis=-1).reshape(-1, 1)
#    lp = lp / np.max(np.abs(lp))
#    
#    hp = hp - hp.mean(axis=-1).reshape(-1, 1)
#    hp = hp / np.max(np.abs(hp))
    
    
    return lp, hp

#%% Testing output spectrum

##x = np.random.normal(size=[1, 512])
##yl, yh = band_split(x)
#
#x = data_loader(1, 512)
#yl, yh = data_parser(x, 512, 1, overlap=0, apply_mask=False)
#
#print('yl.shape', yl.shape)
#print('yh.shape', yh.shape)
#plt.figure()
#plt.plot(yl[0,:])
#plt.title('Sxx_L')
#plt.figure()
#plt.plot(yh[0,:])
#plt.title('Sxx_H')

