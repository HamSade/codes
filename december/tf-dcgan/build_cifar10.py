#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:20:03 2018
@author: hsadeghi
"""

import pickle

load_path  = "/vol/grid-solar/sgeusers/hsadeghi/databases/cifar/cifar-10-batches-py/"
save_path = "/vol/grid-solar/sgeusers/hsadeghi/databases/cifar/cifar/" 

batch_label = {}
labels = {}
data ={}

for i in range(1,6):
    
    file  = load_path  + "data_batch_" + str(i)
      
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    
        #dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    
        batch_label[i] = dic[b"batch_label"]
        labels[i] = dic[b"labels"]
        print("labels length = ", len(labels[i]))
        
        data[i] = dic[b"data"]
        
        print("data shape = ", data[i].shape )
        
        
        
        
        print(batch_label[i])