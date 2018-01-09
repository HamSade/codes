"""
Created on Sat Apr  8 21:47:24 2017

@author: hsadeghi
"""

import tensorflow as tf


#%%
def binary_quantizer(x, mode):
    return x
#    return 2.0 * tf.round( (x +  1.0) / 2.0 ) - 1.0
##    return tf.fake_quant_with_min_max_args(x, min=-1.0, max=1.0)

 
#%%    
#def binary_quantizer(x, mode):
#        # making sure data is between [-1, 1]
##        x = tf.maximum(x, -1.)
##        x = tf.minimum(x, 1.)
##        x_q = tf.cond( tf.less( tf.cast(mode, tf.float32) , 0.5),
#        x_q = tf.cond( tf.less( mode , 0.5),
#                           lambda: training_pass(x),
#                           lambda: test_pass(x) )
#        return x_q
#    
#def training_pass(x):
#    g_x = -1. + 2. * tf.ceil( tf.add( (1.+x)/2. ,  -tf.random_uniform(tf.shape(x))   ))
#    return tf.add( x, tf.stop_gradient( tf.add(-x, g_x) ) )                                                                                                                            
#    
#def test_pass(x):
#    # if the input is not in [-1, 1], then??? RESOLVED ON JULY 4 !!!!!!!
#    return 2.0 * tf.round( (x +  1.0) / 2.0 ) - 1.0  
##    