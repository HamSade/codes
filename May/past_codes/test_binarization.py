#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:46:24 2017

@author: hsadeghi
"""
from __future__ import division

import scipy.io as si
import numpy as np
import tensorflow as tf

from binary import float_to_one_hot, one_hot_to_float


#%%
path_name = '/home/hsadeghi/Downloads/Data/'

def data_loader(file_name):
    
    
#    file_name = '/local/scratch/PDA/my_data/'+ file_name #on fujiama
    file_name = path_name + file_name #on servers
    
    mat = si.loadmat(file_name)  # 'com_concat_signal.mat'
#    fs=np.asscalar(mat['fs_new']);
    
    data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
    data=np.array(data); 
    n_data=data.shape[1];

    return data, n_data

#%%

x_orig, n = data_loader('com_concat_signal_1.mat')

x= x_orig[0, 0:100000]

#%% #################################
#########   Testing functions   ######
#####################################
sess=tf.Session()

#b=dec_to_bin(0.99*tf.ones([2,1]), 4);

#a=[0.001 ,-0.1 ,0.5, -.5, .99, -.98]

a=tf.random_uniform([2,3])


c= float_to_one_hot(a, 8)
d= one_hot_to_float(c, 8)

print( sess.run(c))
print( sess.run(d))

#print( sess.run( tf.reduce_sum(tf.add(a,-d))/tf.to_float(tf.shape(a)[-1])  ))

#sess.close()











