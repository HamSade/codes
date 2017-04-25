#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:55:35 2017

@author: hsadeghi
"""

import tensorflow as tf

#def rnn_layer(x, num_neurons, previous_state):
#
#    lstm = tf.contrib.rnn.BasicLSTMCell(num_neurons)
#    
#    
#    for i in range(x.get_shape[1]):
#        output, state = lstm(x[:, i], previous_state)
#        
#    return output, state


def rnn_layer(x, previous_state, w, b):
     
     # x is [num_batches , input_dim]
     # weights is [ 4, in_dim + num_neurons , num_neurons]
     # biases is [4, num_neorons]
     # previous_state is [2 , batch_size , num_neurons] includes both h & c
       
     h= previous_state[0]
     c= previous_state[1]
     
     output=tf.zeros(x.get_shape)
     
     for idx in range(x.get_shape()[0].value):
         
         # xx=tf.transpose(x)
         Th = tf.matmul( w, tf.concat( [h, tf.transpose(x[idx,:]) ], 0) )
         
         i = tf.sigmoid( tf.add( Th[0,:] , b[0,:]) )
         f = tf.sigmoid( tf.add( Th[1,:] , b[1,:]) )
         o = tf.sigmoid( tf.add( Th[2,:] , b[2,:]) )
         g = tf.sigmoid( tf.add( Th[3,:] , b[3,:]) )

         c= tf.add ( tf.multiply(f, c) , tf.multiply(i, g) )
         
         h= tf.multiply ( o, tf.tanh(c) )
         
         state = tf.stack([h,c])
             
         output[idx,:]=h
     
     return output, state
 
    
    
    #def rnn_layer(x, n_neurons, init_state):
#    
##    print(x.get_shape()[1].value)
#    
#
##    x=tf.reshape(x,shape=[batch_size,1,x.get_shape()[1].value])
##    x = tf.transpose(x, [1, 0, 2])
##    x = tf.reshape(x, shape=[-1, x.get_shape()[2].value])
#    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
##    x = tf.split(x, 1, 0)
#    
##    print(x.get_shape())
#    
#    y = tflearn.layers.recurrent.lstm (x, n_neurons, activation='tanh',
#                                   inner_activation='sigmoid', dropout=dropout_p,
#                                   bias=True, 
#                                   weights_init=tf.random_normal,
#                                   forget_bias=1.0, return_seq=True,
#                                   return_state=True, initial_state=init_state)
#                                                   
#    output = tf.reshape(tf.concat(1, y[0]), [-1, n_neurons])
#    state= y[1]
#                 
#    return output, state
#    

#def rnn_layer(x, n_neurons, init_state):
#    
#    
##    x=tf.transpose(x, [1,0,2])
##    x = tf.transpose(x, [1, 0, 2])
##    x = tf.reshape(x, shape=[-1, x.get_shape()[2].value])
#    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
##    x = tf.split(x, batch_size, 2)
#       
#    #    init_state = lstm.zero_state(batch_size, tf.float32)
#        
##    x=batch_normalization(x);
#
##    x=tf.transpose(x, perm=[1, 0, 2])
#
#    lstm = tf.contrib.rnn.LSTMCell(n_neurons, state_is_tuple=False)
#    lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=dropout_p,
#                                         output_keep_prob=dropout_p)
#
##    output, state = tf.contrib.rnn.static_rnn(lstm, x, initial_state=init_state)   
#    output, state = tf.contrib.rnn.static_rnn(lstm, x, initial_state=init_state) 
#        
#    return output[-1], state
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    def rnn_layer(x, previous_state, w, b):
#     
#     # x is [num_batches , input_dim]
#     # weights is [ 4, in_dim + num_neurons , num_neurons]
#     # biases is [4, num_neorons]
#     # previous_state is [2 , batch_size , num_neurons] includes both h & c
#       
#     h_past= previous_state[0];
#     c_past= previous_state[1];
#     
#     # xx=tf.transpose(x)
#     Th_i = tf.matmul( tf.concat([x, h_past ], 1), w[0])
#     Th_f = tf.matmul( tf.concat([x, h_past ], 1), w[1])
#     Th_o = tf.matmul( tf.concat([x, h_past ], 1), w[2])
#     Th_g = tf.matmul( tf.concat([x, h_past ], 1), w[3])
#     
#     n_time_samples = x.get_shape()[0].value   # equals batch_size in the training phase
#     
#     b_i=tf.reshape( tf.tile (b[0,:],[n_time_samples]), [n_time_samples, -1]) 
#     b_o=tf.reshape( tf.tile (b[1,:],[n_time_samples]), [n_time_samples, -1])
#     b_f=tf.reshape( tf.tile (b[2,:],[n_time_samples]), [n_time_samples, -1])
#     b_g=tf.reshape( tf.tile (b[3,:],[n_time_samples]), [n_time_samples, -1])
#
#     
#     i = tf.add( b_i, tf.sigmoid( Th_i )) # These will be all column vectors
#     o = tf.add( b_o, tf.sigmoid( Th_f ))
#     f = tf.add( b_f, tf.sigmoid( Th_o ))
#     g = tf.add( b_g, tf.tanh(    Th_g ))
#     
#     c= tf.add ( tf.multiply(f, c_past) , tf.multiply(i, g) )
#     
#     h= tf.multiply ( o, tf.tanh(c) )
#     
#     state = tf.stack([h,c])
#         
#     output=h
#     
#     return output, state