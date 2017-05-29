#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:16:33 2017

@author: hsadeghi
"""
import tensorflow as tf
import numpy as np
import scipy.io as si 
from tensorflow.contrib import ffmpeg


path_name_clean = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/clean_trainset_wav/'
path_name_noisy = '/vol/grid-solar/sgeusers/hsadeghi/segan_data/noisy_trainset_wav/'















#%%
#def data_loader(input_dim=1024, wavetype='clean'):
#    
#    if wavetype == 'clean':
#        path_name = path_name_clean
#    else:
#        path_name = path_name_noisy
#        
#    ##########################3    
#    file_list = tf.train.match_filenames_once(path_name+"*.wav")
#    
#    filename_queue = tf.train.string_input_producer(file_list)
#
#    audio_binary = tf.read_file(filename_queue)
#    
#    waveform = ffmpeg.decode_audio(audio_binary,
#                                   file_format='wav',
#                                   samples_per_second=16000)
##    serialized_inout
##    
##    for i in waveform:
#        
#
#
##    reader = tf.TFRecordReader()
##    _, serialized_example = reader.read(filename_queue)
##    features = tf.parse_single_example(
##            serialized_example,
##            features={
##                'wav_raw': tf.FixedLenFeature([], tf.string),
##                'noisy_raw': tf.FixedLenFeature([], tf.string),
##            })
##    wave = tf.decode_raw(features['wav_raw'], tf.int32)
##    wave.set_shape(canvas_size) #reshaping to the desired shape
##    
##    # projecting to [-1,1]
##    wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
##    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
##    noisy.set_shape(canvas_size)
##    noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.
##
##    if preemph > 0:
##        wave = tf.cast(pre_emph(wave, preemph), tf.float32)
##        noisy = tf.cast(pre_emph(noisy, preemph), tf.float32)
#
#    return waveform #wave, noisy



#%%

def parse_wav(data, input_dim):
    
    data=np.array(data); 

    n_data=data.shape[1];
    
    n_data = int(2 * n_data / input_dim) - 1
          
    formatted_data = np.zeros([n_data, input_dim])
      
    for  i in range(n_data):
        formatted_data[i,:]= data[0, i * int(0.5*input_dim) : i* int(0.5*input_dim) + input_dim];  # 50% overlap 
    
    return formatted_data, n_data


#%%
def add_noise(x, std=0.002):
    
    x=np.array(x)
    y =np.add(  x , np.random.normal(loc=0.0, scale=std, size=x.shape))
    
    return y


#%% test tf file reading


a=  data_loader()


#%% test

#wav_name= path_name + 'com_concat_signal_1.mat'

#import tensorflow as tf
#
#sess=tf.Session()
#
#a=tf.ones([2,3])
##a=[1,2,3]
##
#b= add_noise(a, 0.002)
##
##
#print(sess.run(b))
#
#sess.close()


