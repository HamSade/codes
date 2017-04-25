"""
Created on Sat Apr  8 21:47:24 2017

@author: hsadeghi
"""

#import tensorflow as tf


class binary_quantizer():
    
    def __init__(self, tf):
               
        self.tf= tf
        
        
    def __call__(self, x, mode):  #, self.tf):
    
        x_q = self.tf.cond( self.tf.less( self.tf.cast(mode, self.tf.float32) , 0.5),
                           lambda: self.training_pass(x),
                           lambda: self.test_pass(x) )
        
        return x_q
    
    def training_pass(self, x):
        
        g_x = -1. + 2. * self.tf.ceil( self.tf.add( (1.+x)/2. ,  -self.tf.random_uniform(self.tf.shape(x))   ))
        
        return self.tf.add( x, self.tf.stop_gradient( self.tf.add(-x, g_x) ) )                                                                                                                            
    
    def test_pass(self, x):
        
        # if the input is not in [-1, 1], then???
        return 2.0 * self.tf.round( (x +  1.0) / 2.0 ) - 1.0  # b^{inf} in Toderici's paper
                                                    # epsilon = 1e-8 on my machine i.e. test_pass(1e-8) = -1 !!
    