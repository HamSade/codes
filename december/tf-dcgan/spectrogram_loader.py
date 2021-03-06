#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:07:57 2017

@author: hsadeghi
"""

#%%
#import tensorflow as tf
import numpy as np
import scipy.io as si 
import matplotlib.pyplot as plt

#%%
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_spec_clean_16k/'

fs = 16000
plot = 0

#%%
def data_loader(file_ind, input_dim): 

    file_name = path_name + 'clean_spec_{}.mat'.format(file_ind)
    mat = si.loadmat(file_name)  
    Sxx = mat['Sxx']
#    nperseg = mat['nperseg']
#    noverlap = mat['noverlap']
    Sxx = np.array(Sxx)
    return Sxx
    
#%%
def data_parser(Sxx, input_dim, batch_size, down_ratio=4):
   
    nperseg = int( 32 /1000. * 16000 )
#    noverlap = nperseg * 3 // 4
    
    num_all_bins = Sxx.shape[-1]
    num_required_bins = int( batch_size * input_dim / nperseg) * 4 ;  #4 = nperseg/S    
    rand_ind = np.random.randint( 0 , num_all_bins - num_required_bins )

    Sxx = Sxx [:, rand_ind : rand_ind + num_required_bins] 
    
    # normalizing among all batch frames
#    maxi =  np.max(np.abs(Sxx))
#    Sxx = Sxx / maxi * 0.9
       
    # Splitting into batches
    Sxx = np.asarray(np.array_split(Sxx, batch_size, axis = -1))
    
    # Downsampling the frequency content
#    Sxx = Sxx[:, :-1:down_ratio, ::2]
    Sxx = Sxx[:, :-1:down_ratio, :]
    
#    Sxx =  Sxx[:, 2:-2, 2:-2]  #to make it 28 x 28
        
    # nomalizing each frame inside batch
    maxi = np.amax(abs(Sxx), axis=(-1,-2), keepdims=True)
    Sxx = Sxx / maxi * 0.9

    # MAKE OUTPUT 4-DIM
#    Sxx = np.reshape(Sxx, (batch_size, Sxx.shape[1], Sxx.shape[2], 1)#    
    
    # make output RGB with repitition
    Sxx = np.reshape(Sxx, (batch_size, Sxx.shape[1], Sxx.shape[2], 1))
    Sxx = np.repeat(Sxx, 3, axis = -1)
    
     # make output RGB with adding zeros
#    zeros = np.zeros_like(Sxx)
#    Sxx = np.stack((Sxx, zeros, zeros), axis=-1)
#    
    Sxx = np.float32(Sxx)
    
    return Sxx#, maxi

#%%

if plot==1:
    
    input_dim = 2 ** 13
    x = data_loader(0, input_dim)
    Sxx = data_parser(x, input_dim, 128)
    
    print('Sxx.shape after split', Sxx.shape)
#    print("max(Sxx)", np.max(Sxx[0]))
    
    plt.figure()
    img = np.squeeze(Sxx[0,:,:,0])
#    img = img + abs(np.min(img))
#    img = img/np.max(img)
    plt.imshow(img)
    print('min(img)', np.min(img))
    print('max(img)', np.max(img))
    
    
#    plt.figure()
#    plt.pcolormesh(np.squeeze(img[:,:,0]))
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.show()
