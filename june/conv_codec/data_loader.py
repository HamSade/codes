#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:01:33 2017

@author: hsadeghi
"""

import tensorflow as tf
import numpy as np
import scipy.io as si 
import scipy.signal as ss 

#import mdct

#%%
#path_name = '/vol/grid-solar/sgeusers/hsadeghi/data/'
path_name = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/mat_clean_16k/'

fs = 16000
fc = 2000
wn = fc/ (fs/2)  #cutoff for speech    

#%%
def data_loader(file_ind, input_dim, mdct_indicator=True):
    file_name = path_name + 'clean_{}.mat'.format(file_ind) 
#    file_name = path_name + 'com_concat_signal_{}.mat'.format(file_ind) 
    
    mat = si.loadmat(file_name)  
    #    fs=np.asscalar(mat['fs_new']);
    data = mat['concat_wav']
    data = np.array(data) #data.shape = (1, 47456861)
    # clip length to be a multiple of 2048
    n_data = data.shape[1];
    n_data = n_data - n_data % input_dim
    data = data[0,0:n_data] # data.shape = (47456768,)
    # apply mdct
    if mdct_indicator:
        data = apply_mdct(data, framelength=input_dim * 2)
        #mapping coeffs to [-1,1]
        mean_value = np.mean(data)
        data = data - mean_value
        max_value = np.max(np.abs(data))
        data = data / max_value
        return data, max_value, mean_value
    else:
        return data
 
#%%
def band_split(x):
    # input_dim is : batch_size, input_dim + overlap * 2
    x_f = np.fft.fft(x, axis=-1)   
    
    length = x.shape[-1]   
    nc = int(wn * length / 2)  #index of cutoff
    mask_l = np.ones([length])
    mask_l [ nc : length- nc + 1 ] = 0  # +1 comes from the middle frequency fft(length/2 (zero indexing))
    mask_l.astype(np.complex128)
    lpf = np.multiply(x_f, mask_l);
#    mask_h = np.ones([length]) - mask_l;
    hpf = np.subtract(x_f, lpf);
    
    hpf_l = np.zeros([hpf.shape[0], length], dtype=np.complex128)
    hpf_l [:, 0: int(length/2) - nc + 1 ] = hpf[ :, nc : int(length/2)+1]
    hpf_l [:, int(length/2) + nc: length]    = hpf[ :, int(length/2): length - nc ]
    
    lowpass = np.fft.ifft(lpf); 
    highpass = np.fft.ifft(hpf_l);
    return lowpass.astype(np.float64), highpass.astype(np.float64)    

#%%
def band_merge(lowpass, highpass):
    lpf = np.fft.fft(lowpass, axis=-1) 
    hpf = np.fft.fft(highpass, axis=-1)
    
    length = hpf.shape[-1]
    nc = int(wn * length / 2)
    hpf_h = np.zeros([hpf.shape[0], length], dtype=np.complex128)
    
    hpf_h[ :, nc : int(length/2)+1] = hpf [:, 0: int(length/2) - nc + 1 ]
    hpf_h[ :, int(length/2): length - nc ] = hpf [:, int(length/2) + nc: length]
    
    x_f = np.add(lpf, hpf_h)
    x = np.fft.ifft(x_f)
    return x.astype(np.float64)

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

#    lowpass, highpass = band_split(parsed_data)
#    return lowpass, highpass

    # 3-band approach
    l, h = band_split(parsed_data)
    hl, hh = band_split(h)
    return l, hl, hh

#    # 4-band approach
#    l, h = band_split(parsed_data)
#    hl, hh = band_split(h)
#    hhl, hhh = band_split(hh)
#    return l, hl, hhl, hhh   

#%%

#SHOULD BE MODIFIED FOR OVERLAP! DONT FORGETTTTTTTTTTTTTTTTTTT

def data_parser_2(data, batch_size, overlap=0, apply_mask=False):
    
    n_data = data.shape[1];
    rand_ind = np.random.randint( 0 + overlap, n_data -  (batch_size+overlap))
    parsed_data = data[ : , rand_ind - overlap : rand_ind + batch_size + overlap ]
    parsed_data = np.transpose(parsed_data)
    return parsed_data

#%%
def apply_mdct(x, framelength=512):
#    x = np.reshape(x,[-1,1])
    x = mdct.fast.mdct(x, framelength=framelength) 
#    x = np.reshape(x,[-1,1])
    return x
#%%
def apply_imdct(x, framelength=512):
#    x = np.reshape(x,[framelength/2,-1])
    x = mdct.fast.imdct(x, framelength=framelength * 2)
#    x = np.reshape(x,[-1,1])
    return x

#%%
def pre_emph(x, coeff=0.95):
    x=tf.to_float(x)
#    x = x.astype(np.float32)
    x0 = tf.reshape(x[:,0], [-1,1]) 
    diff = x[:, 1:] - coeff * x[:, :-1]
    concat = tf.concat([x0, diff], axis=1)
    return concat

#%%
def de_emph(y, coeff=0.95):
    y = tf.to_float(y)
    if coeff <= 0:
        return y
    x = y[:, 0]
    x= tf.reshape(x , [-1,1])    
    for n in range(1, y.get_shape().as_list()[-1]):
        new_col = tf.add(coeff * x[:, n - 1] , y[:,n])
        new_col = tf.reshape(new_col , [-1,1])
        x = tf.concat([x, new_col],axis=1)
    return x

#%% Test band split and merge
#import matplotlib.pyplot as plt
#x=np.random.normal(size=[128, 512])
#l, h = band_split(x)
#x_ = band_merge(l, h)
#e=np.subtract(x,x_)
##plt.figure()
#plt.plot(np.sum(np.abs(e)/x.shape[-1], axis=0))

#%%
#def band_split(x):
#    # input_dim is : batch_size, input_dim + overlap * 2
#    b_low, a_low = ss.butter(50, wn)   # Butterworth
#    lowpass = ss.lfilter(b_low, a_low, x, axis=-1);
#    highpass = np.subtract(x,  lowpass)  # this should be downsampled by 3/4
##    lowpass = ss.decimate(lowpass, 4) #since only 1/4 of the bandwith
##    modulator = np.sin( 2 * np.pi * fc / (fs/2)  * np.arange(x.shape[-1]) )
#    modulator = np.power(-1,np.arange(lowpass.shape[-1]))
#    highpass = np.multiply( highpass, modulator)
#    highpass = ss.lfilter(b_low, a_low, highpass, axis=-1)
#    return lowpass, highpass    
#
##%%
#def band_merge(lowpass, highpass):
##    lowpass = ss.resample(lowpass, lowpass.shape[-1]*4, axis=-1)
##    lowpass = ss.lfilter(b, a, lowpass, axis=-1);      
##    modulator = np.sin( 2 * np.pi * fc / (fs/2)  * np.arange(lowpass.shape[-1]) ) 
#    modulator = np.power(-1,np.arange(lowpass.shape[-1]))
#    highpass = np.multiply( highpass, modulator)
#    b_high, a_high = ss.butter(20, wn)
##    b_high, a_high = ss.butter(50, wn, btype='highpass')
#    highpass = ss.lfilter(b_high, a_high, highpass, axis=-1);
#    x = np.add(lowpass, highpass) 
#    return x 

#%% Test band split and merge
#import matplotlib.pyplot as plt
#
#x=np.random.normal(size=[128, 512])
#l, h  = band_split(x)
#plt.plot(h[100,:])
#x_= band_merge(l, h)
#e=np.subtract(x,x_)
#plt.figure()
#plt.plot(np.sum(np.abs(e)/x.shape[-1], axis=0))


#%% def band_merge(lowpass, highpass):
#    #numpy version
#    lowpass = ss.resample(lowpass, lowpass.shape[-1]*4, axis=-1)
#    lowpass = ss.lfilter(b, a, lowpass, axis=-1);      
#    x = np.add(lowpass, highpass)  # We are omitting HPF  for highpass band  
#    return x 
 

#%% test mdct and imdct
#a_,a = data_loader(5, 512)
#a=np.random.normal(size=[512*128])
##a_=a
#a_=apply_mdct(a, 512 *2)
#a__=apply_imdct(a_, 512)
#print('error',np.sum(np.abs(np.subtract(a,a__))))

#%%
#def data_parser(data, input_dim, batch_size, overlap=False):
#    
#    data = np.reshape(data, [1,-1])
#    n_data=data.shape[1];
#    parsed_data = np.zeros([batch_size, input_dim])
#    
#    if overlap:
#        rand_ind = np.random.randint(0, n_data - int(2/3.* input_dim * batch_size) - 1)
#        for  i in range(batch_size):    
#            parsed_data[i,:]= data[0, rand_ind + i * int(0.5*input_dim) :\
#                          rand_ind + i* int(0.5*input_dim) + input_dim];  # 50% overlap
#    else:
#        rand_ind = np.random.randint(0, n_data - input_dim * batch_size)
#        ### Slow parsing
##        for  i in range(batch_size): 
##             parsed_data[i,:]= data[0, rand_ind + i * input_dim :\
##                       rand_ind + (i + 1) * input_dim];  # 0 overlap 
#        ### Fast parsing
#        vec_data = data [0, rand_ind: rand_ind + batch_size * input_dim]                
#        parsed_data = np.reshape( vec_data, [batch_size, input_dim])
#
#    return parsed_data

#%%
#def data_parser(data, input_dim, batch_size, overlap=0, apply_mask=False):
#    
#    data = np.reshape(data, [1,-1])
#    n_data=data.shape[1];
#    parsed_data = np.zeros([batch_size, input_dim + overlap * 2 ])
#
#    rand_ind = np.random.randint( 0 , n_data - batch_size * (input_dim + overlap * 2))
#    
#    # Trapezoidial tapered mask
##    zeros = np.zeros(overlap/2)
##    ramp_1 = np.array(range(overlap/2))/float(overlap/2-1)
##    flat = np.ones(input_dim)
##    ramp_2 = 1-ramp_1
##    mask = np.concatenate([zeros, ramp_1, flat, ramp_2, zeros])
#    
#    # Rect mask
#    zeros = np.zeros(int(overlap))
#    flat = np.ones(input_dim)
#    mask = np.concatenate([zeros, flat, zeros])
#
#    if apply_mask:
#        for  i in range(batch_size): 
#            data_window = data[0, rand_ind + i * (input_dim) :\
#                                      rand_ind + (i+1) * (input_dim) + overlap*2 ]  
#            parsed_data[i, 0:input_dim + overlap * 2] = np.multiply(mask, data_window)
#        
#    else:       
#        for  i in range(batch_size): 
#            #trapezoid
#    #        data_window = data[0, rand_ind +  i * (input_dim + overlap/2) :\
#    #                              rand_ind + (i+1) * (input_dim + overlap/2) + 3*overlap/2 ]
#    #        parsed_data[i, 0:input_dim + overlap * 2]= np.multiply( mask, data_window)
#            data_window = data[0, rand_ind + i * (input_dim) :\
#                                      rand_ind + (i+1) * (input_dim) + overlap*2 ]  
#            parsed_data[i, 0:input_dim + overlap * 2] = data_window
#    return parsed_data

#%%
#def pre_emph(x, coeff=0.95):
#    x=tf.to_float(x)
##    x = x.astype(np.float32)
#    x0 = np.reshape(x[:,0], [-1,1]) 
#    diff = x[:, 1:] - coeff * x[:, :-1]
#    concat = np.concatenate((x0, diff), axis=1)
#    return concat

#%%
#def de_emph(y, len_y, coeff=0.95):
#    y = y.astype(np.float32)
#    if coeff <= 0:
#        return y
#    x = y[:, 0]
#    x= np.reshape(x , [-1,1])    
#    for n in range(1, len_y):
#        new_col = np.add(coeff * x[:, n - 1] , y[:,n])
#        new_col = np.reshape(new_col , [-1,1])
#        x = np.concatenate([x, new_col],1)
#    return x
#%% test
#x=[[1,2],[3,4]]
#x=np.array(x)
#x_= pre_emph(x)
#y= de_emph(x_)
###
##print('x_',x_)
##print('y',y)
#sess=tf.Session()
##
#print(sess.run(x_))
#print(sess.run(y))

#%%
#def pre_emph(x, coeff=0.95):
#    x=tf.to_float(x);
#    x0 = tf.reshape(x[:,0], [-1,1])
#    diff = x[:,1:] - coeff * x[:,:-1]
#    concat = tf.concat([x0, diff], axis=1)
#    return concat
#
##%%
#def de_emph(y, coeff=0.95):
#    if coeff <= 0:
#        return y
##    x = np.zeros(y.shape, dtype=np.float32)
##    x[0] = y[0]
#    x = y[:, 0]
#    for n in range(1, y.shape[0], 1):
#        x[n] = coeff * x[n - 1] + y[n]
#    return x


#%% testing de_emph

#start_time = time.time()
#data = data_loader(1)
#print('---data loading = {} seconds--'.format(time.time()-start_time))
#
#start_time = time.time()
#data = data_parser(data, 2**14, 128)
#print('---data parsing = {} seconds--'.format(time.time()-start_time))
