#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:29:41 2017

@author: hsadeghi
"""

from os import walk
import time

path_clean = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/clean_trainset_wav_16k/'
path_noisy = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/noisy_trainset_wav_16k/'

#%% file lists

start_time = time.time()

file_names_clean = []
file_names_noisy = []

for (dirpath, dirnames, filenames) in walk(path_clean):
    file_names_clean.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(path_noisy):
    file_names_noisy.extend(filenames)
    break

print("--- %s seconds ---" % (time.time() - start_time))

#%%

f  = open('clean_list.txt', "w") 

for i in range(len(file_names_clean)):
    f.write(str(file_names_clean[i]))
    f.write('\n')


f.close()


f  = open('noisy_list.txt', "w") 

for i in range(len(file_names_noisy)):
    f.write(str(file_names_noisy[i]))
    f.write('\n')
    
f.close()


