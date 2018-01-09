#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:56:28 2017

@author: hsadeghi
"""

from spectrogram_loader import data_loader, data_parser
import numpy as np
import matplotlib.pyplot as plt

#%%
num_files = 1
batch_size = 1

print("Loading data started") 
input_dim = 2**13
data = {}
for i in range(num_files):
    data[i] = data_loader(i, input_dim)

print("Loading data finished") 

#%%
def inf_train_gen(data=data):
    while True:
        rand_ind = np.random.randint(0, num_files)
        yield data_parser(data[rand_ind], input_dim, batch_size)
            
loader = inf_train_gen()

#%%

for i in range(5):
    
    Sxx = next(loader)

    print('Sxx.shape after split', Sxx.shape)
    
    plt.figure()
    img = np.squeeze(Sxx[0,:,:,0])
#    img = img + abs(np.min(img))
#    img = img/np.max(img)
    plt.imshow(img)
    print('min(img)', np.min(img))
    print('max(img)', np.max(img))
    
#    plt.figure()
#    plt.pcolormesh(Sxx[0])
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.show()
#    #    plt.pause(1)
#    
#    Sxx = next(loader)
#    
#    print('Sxx.shape after split', [len(Sxx), Sxx[0].shape])
#    print("max(Sxx)", np.max(Sxx[0]))
#    
#    plt.figure()
#    plt.pcolormesh(Sxx[0])
#    plt.ylabel('Frequency [Hz]')
#    plt.xlabel('Time [sec]')
#    plt.show()


