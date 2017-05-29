#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:16:33 2017

@author: hsadeghi
"""
import tensorflow as tf
import numpy as np
import scipy.io as si 


#path_name = '/vol/grid-solar/sgeusers/hsadeghi/data/'
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/'

def data_loader(file_name, input_dim, overlap):
    
    
#    file_name = '/local/scratch/PDA/my_data/'+ file_name #on fujiama
    file_name = path_name + file_name #on servers
    
    mat = si.loadmat(file_name)  # 'clean_.mat'
#    fs=np.asscalar(mat['fs_new']);
    
    data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
    data=np.array(data); 
    n_data=data.shape[1];
    
    if overlap:
        n_data = int(2 * n_data / input_dim) - 1
    else:
        n_data = int(n_data / input_dim)

    formatted_data = np.zeros([n_data, input_dim])
      
    for  i in range(n_data):
        if overlap:
            formatted_data[i,:]= data[0, i * int(0.5*input_dim) : i* int(0.5*input_dim) + input_dim];  # 50% overlap
        else:
            formatted_data[i,:]= data[0, i * input_dim : (i + 1) * input_dim];  # 0 overlap 
    
    return formatted_data, n_data

#%%
def pre_emph(x, coeff=0.95):
#    x = tf.cast(x, tf.float32)
    x0 = tf.reshape(x[:,0], [-1,1]) 
    diff = x[:, 1:] - coeff * x[:, :-1]
    concat = tf.concat([x0, diff], 1)
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


#%% testing de_emph

#sess=tf.Session()
#
#a = tf.ones([2,10])
##
#b = de_emph(a, 1024)
##
#
#print(sess.run(b))
#
#print(end-start)
#sess.close()


# with np

#a = np.random.normal(size=[2,3])
#
#aa= sess.run(pre_emph(a))
#
#b = de_emph(aa, 3)
#
#print(a,b)
#
#
#sess.close()