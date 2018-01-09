#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 09:50:39 2017

@author: hsadeghi
"""

import dcgan as dc
import tensorflow as tf
import numpy as np
#from time import time
from spectrogram_loader import data_loader, data_parser
import matplotlib.pyplot as plt

import scipy.io as sio 

#%% Parameters
batch_size = 128
num_steps = int(100000)
display_step = 10

#%% Loading data
print("Loading data started") 
input_dim = 2**13
data = {}
for i in range(10):
    data[i] = data_loader(i, input_dim)

print("Loading finished")

#%%
def inf_train_gen(data=data):
    while True:
        rand_ind = np.random.randint(0, 10)
        yield data_parser(data[rand_ind], input_dim, batch_size)
        
loader = inf_train_gen(data)

#%% Model

dcgan = dc.DCGAN()
#train_images = <images batch>
train_images  = next(loader)

losses = dcgan.loss(train_images)
train_op = dcgan.train(losses)

gen_vec = []
disc_vec = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(num_steps):
        _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]])
        
        # save cost trace
        gen_vec.append(g_loss_value)
        disc_vec.append(d_loss_value)
        
        # printing costs
        print('iteration = ', step, '  disc_cost = ', d_loss_value, '  gen_cost = ', g_loss_value )
        
        if step % (display_step) == 0:
                plt.figure(1)
                
                if np.abs(d_loss_value)<1.5:
                    plt.plot(step, (10**4) * d_loss_value, 'r*') 
#                    plt.hold(True)
                    plt.pause(0.01)
                  
                if np.abs(g_loss_value)<1.5:
                    plt.plot(step, (10**4) * g_loss_value, 'b.') 
#                    plt.hold(True)
                    plt.pause(0.01)
                    
    # save trained variables
    
    #%% Saving model
    saver = tf.train.Saver()
    save_path = saver.save(sess, '/vol/grid-solar/sgeusers/hsadeghi/research_results/tf-dcgan/saved_model/model.ckpt')
    print("Model saved in file: %s" % save_path)
    
    #%% Writing costs
    training_costs={};    
    training_costs['coder_cost'] = gen_vec
    training_costs['disc_cost'] = disc_vec
    
    #save_path = "events_training.mat"
    save_path = "/vol/grid-solar/sgeusers/hsadeghi/research_results/tf-dcgan/saved_model/events_training.mat"
    sio.savemat(save_path, training_costs);
    sess.close()
    print("Training events saved in file: %s" % save_path)
        
    
