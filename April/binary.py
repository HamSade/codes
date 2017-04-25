#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:22:56 2017

@author: hsadeghi
"""

#%%
#import tensorflow as tf

from binary_quantizer import binary_quantizer


#%%

def bitwise_layer(x, width, W, b, tf, mode):
    
    # W must beof size [width+1, width]
    
#    b_q = binary_quantizer(tf)  #binary quantizer initialization
    
    
#    out_temp = tf.zeros([ tf.shape(x)[0], width]);
    
#    xo =x;  # considering x is batch_size by 1
    
    out_temp = tf.matmul ( x, W[0]) + b[0]
    
    xo = tf.concat([x, out_temp], axis=1)
    
    for i in range(width-1):
        
        out_temp = tf.concat([out_temp,  tf.matmul( xo, W[i+1])+ b[i+1] ], axis=1)    

#        out_temp = tf.concat([out_temp, out_temp], axis=1)

        xo = tf.concat([x, out_temp], axis=1)
    
                
        

    output = tf.stack(out_temp);
    
    return output
#    return b_q (output, mode)


#%%

def bitwise_layer_2(x, width, W, b, tf, mode):
    
    # W must beof size [width+1, width]
    
#    b_q = binary_quantizer(tf)  #binary quantizer initialization
    
    
#    out_temp = tf.zeros([ tf.shape(x)[0], width]);
    
#    xo =x;  # considering x is batch_size by 1
    
    out_temp = tf.matmul ( x, W[0]) + b[0]
    
    xo = tf.concat([x, out_temp], axis=1)
    
    for i in range(width-1):
        
        out_temp = tf.concat([out_temp,  tf.matmul( xo, W[i+1])+ b[i+1] ], axis=1)    

#        out_temp = tf.concat([out_temp, out_temp], axis=1)

        xo = tf.concat([x, tf.reshape( out_temp[:,-1] , [-1, 1] ) ], axis=1)
    
                
        

    output = tf.stack(out_temp);
    
    return output




#%%
def binary_cost(o, t, tf):
    
    return 0.5 * tf.reduce_sum( 1. - tf.multiply(o, t), 1)
    
#%%

# float to n-bit binary format conversion for inputs in [-1,1]

#def dec_to_bin(x, n, tf):
#    
#    x_2 = x;
#    x_b = tf.zeros([tf.shape(x)[0], 1])
#    
#    for _ in range(n):
#        
#        x_2 = tf.cast( tf.cast( 2 * x_2 , tf.int32), tf.float32)  
#
#        x_b = tf.concat ( [x_b, x_2], axis=1) 
#        
##        x_2 = 
#
#    return x_b[:, 1:]




#%% #################################
#########   Testing functions   ######
#####################################

#b=dec_to_bin(0.99*tf.ones([2,1]), 4, tf); sess.run(b)

#%% Testing functions

#import tensorflow as tf
#
## x samples are in [-1, 1]
#x=2*tf.random_uniform([4,1])-1
#width=3; # length of binary represenattion
#
#
#W=[]
##tf.random_normal([1], stddev=0.01)
#
#for i in range(width):
#    W.append(tf.random_normal([i+1,1], stddev=0.01))
#
#b= tf.random_normal([width+1], stddev=0.001)
#
#
#
#
#y= bitwise_layer(x, width, W,b,  tf, 1)
#
#
#sess=tf.Session()
#
#sess.run([x,y])












