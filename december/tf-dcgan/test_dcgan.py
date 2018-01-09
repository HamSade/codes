#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 15:01:46 2017

@author: hsadeghi
"""


import tensorflow as tf
from time import time
import dcgan as dc

#%%
dcgan = dc.DCGAN()
images = dcgan.sample_images()

saver = tf.train.Saver()

with tf.Session() as sess:
    
    # restore trained variables
    start_time = time()
    load_path = "/vol/grid-solar/sgeusers/hsadeghi/research_results/tf-dcgan/"
#    new_saver = tf.train.import_meta_graph(load_path + 'saved_model/model.ckpt.meta')
#    new_saver.restore(sess, load_path + "saved_model/model.ckpt")
    saver.restore(sess, load_path + "saved_model/model.ckpt")
    print("Model restored in {:.0f} seconds".format( time() - start_time) )
    
    generated = sess.run(images)
    
    with open('/vol/grid-solar/sgeusers/hsadeghi/research_results/tf-dcgan/results.jpg', 'wb') as f:
        f.write(generated)
        
        
        
        
        
        
        
        
        
        
        
        
        
        