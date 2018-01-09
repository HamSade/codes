#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:18:30 2017

@author: hsadeghi
"""

#%%
#import tensorflow as tf
import numpy as np
import scipy.io as si 
#import scipy.signal as ss
import matplotlib.pyplot as plt

#%%
path_name = '/vol/grid-solar/sgeusers/hsadeghi/MNIST/mat_mnist/'

plot = 0

#%%
def data_loader(): 

    file_name = path_name + 'database.mat'
    mat = si.loadmat(file_name)  
    images = mat['database']
    labels = mat['labels']
    images = np.array(images)
    labels = np.array(labels)
    return images
    
#%%
def data_parser(images, batch_size):
      
    rand_ind = np.random.randint(0, 60000, batch_size)
    images = images [rand_ind, :, :] 
#    labels = labels[rand_ind]
    # nomalizing each frame inside batch
    maxi = np.amax(images, axis=(-1,-2), keepdims=True)
    images = images / maxi * 0.9    
    return images, maxi

#%%

if plot==1:
    
    images = data_loader()
    images, maxi = data_parser(images, 128)    
    print('Sxx.shape after split', [len(images), images[0].shape])
    print("max(Sxx)", np.max(images[0]))    
    plt.figure()
    plt.imshow(images[0])
#    plt.pcolormesh(images[0])
    plt.show()
    plt.pause(1)
