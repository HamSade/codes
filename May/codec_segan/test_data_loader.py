#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:07:55 2017

@author: hsadeghi
"""

#import tensorflow as tf
#
#from data_loader import read_and_decode
#from __future__ import print_function




f=open('noisy_list.txt', "r")

for i in f:
    print(i)

    
f.close()
    
    
#with open('noisy_list.txt', "r") as f:
#    file_name_noisy = f.read(5)   






#%%

#
#filename = "/vol/grid-solar/sgeusers/hsadeghi/segan_data/segan.tfrecords"
#
#filename_queue = tf.train.string_input_producer([filename])
#
#wave, noisy = read_and_decode(filename_queue, 2**8, preemph=0.95)
#
#
#
#sess=tf.Session()
#
#
#print(sess.run(tf.shape(wave)[0]))
#


#%%



