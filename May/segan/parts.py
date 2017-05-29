#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:19:59 2017

@author: hsadeghi
"""
#%% 

from __future__ import print_function
import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer

#from ops import *
from ops import downconv, prelu, leakyrelu, deconv, nn_deconv, conv1d, gaussian_noise_layer

from bnorm import VBN

#%% Parameters

init_noise_std =  0
init_l1_weight = 100.
batch_size = 100
g_nl = prelu
preemph =  0.95
epoch  = 86

g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]       

kwidth = 31
enc_layers = 7
  
###########  doubtful
bias_deconv = True
bias_downconv = True
bias_D_conv = True

deconv_type = 'nn_deconv'  #Not_sure!!! or could be 'nn_deconv'

   
do_prelu=False



#%%
"""We assume input to generator is [num_batch, inout_dim]"""

def generator(noisy_w, is_ref, z_on=True):
    
    #is_ref : Creation phase
    
    skips = []
    
    h_i = tf.expand_dims(noisy_w, -1)
    

    for layer_idx, layer_depth in enumerate(g_enc_depths):
        
        bias_init = tf.constant_initializer(0.)
        
        h_i_dwn = downconv(h_i, layer_depth, kwidth=31,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_init=bias_init)   
        
        h_i = h_i_dwn
        
        if layer_idx < len(g_enc_depths) - 1:

                    skips.append(h_i)
                                        
        if do_prelu:
            h_i = prelu(h_i) # default: ref=False
        else:
            h_i = leakyrelu(h_i)       
    #end_for :) 
    
    
    # Adding z to c   
    if z_on:
    # random code is fused with intermediate representation
        z = make_z([batch_size, h_i.get_shape().as_list()[1],g_enc_depths[-1]])
        h_i = tf.concat([z, h_i], 2)  
    
    return h_i, skips

     
#%%        
def make_z(shape, mean=0., std=1.):
    
    z = tf.random_normal(shape, mean=mean, stddev=std)
    
    return z        
    

#%%

def decoder(h_i, skips, z=None, z_on=False):
    
    # z+c is called h_i here
    
    g_dec_depths = g_enc_depths[:-1][::-1] + [1]
    
    
    for layer_idx, layer_depth in enumerate(g_dec_depths):
        
        
        h_i_dim = h_i.get_shape().as_list()
                
        out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth] #2 because of skip connections
        
        bias_init=None
        
        #############
        if deconv_type == 'deconv':

            bias_init = tf.constant_initializer(0.)
    
            h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=2,
                             init=tf.truncated_normal_initializer(stddev=0.02),
                             bias_init=bias_init)

        elif deconv_type == 'nn_deconv':

            bias_init = 0.0

            h_i_dcv = nn_deconv(h_i, kwidth=kwidth, dilation=2,
                                init=tf.truncated_normal_initializer(stddev=0.02),
                                bias_init=bias_init)
                        
        h_i = h_i_dcv        
        
        if layer_idx < len(g_dec_depths) - 1:
                    
            if do_prelu:
                        
                h_i = prelu(h_i)

            else:

                h_i = leakyrelu(h_i)
                        
                # fuse skip connection
                skip_ = skips[-(layer_idx + 1)]
                
                h_i = tf.concat([h_i, skip_], 2)

        else:
            h_i = tf.tanh(h_i)
            
    wave = h_i
        
    # Not sure abotu the following
    
    ret_feats = [wave]
    
    if z_on:
        ret_feats.append(z)

    return ret_feats
        

#%% Discriminator params

bias_D_conv = True
canvas_size = 2**14
disc_noise_std = 0.002

d_num_fmaps = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]


#%%
def discriminator(wave_in, reuse=True):
    
      
    hi = wave_in
    
    hi = tf.expand_dims(wave_in, -1)
        
#    batch_size = int(wave_in.get_shape()[0])


    # set up the disc_block function
   
    with tf.variable_scope('d_model') as scope:
        if reuse:
            scope.reuse_variables()

        def disc_block(block_idx, input_, kwidth, nfmaps, bnorm, activation, pooling=2):

                with tf.variable_scope('d_block_{}'.format(block_idx)):

                    if not reuse:
                        print('D block {} input shape: {}'
                              ''.format(block_idx, input_.get_shape()),
                              end=' *** ')

                    bias_init = None

                    if bias_D_conv:
                        if not reuse:
                            print('biasing D conv', end=' *** ')
                        bias_init = tf.constant_initializer(0.)

                    downconv_init = tf.truncated_normal_initializer(stddev=0.02)

##########################################
                    hi_a = downconv(input_, nfmaps, kwidth=kwidth, pool=pooling,
                                    init=downconv_init, bias_init=bias_init)
##########################################                    
                    
                    if not reuse:
                        print('downconved shape: {} '
                              ''.format(hi_a.get_shape()), end=' *** ')
                    
                    if bnorm:
                        if not reuse:
                            print('Applying VBN', end=' *** ')
                        hi_a = vbn(hi_a, 'd_vbn_{}'.format(block_idx))
                    
                    if activation == 'leakyrelu':
                        if not reuse:
                            print('Applying Lrelu', end=' *** ')
                        hi = leakyrelu(hi_a)
                    
                    elif activation == 'relu':
                        if not reuse:
                            print('Applying Relu', end=' *** ')
                        hi = tf.nn.relu(hi_a)
                    
                    else:
                        raise ValueError('Unrecognized activation {} '
                                         'in D'.format(activation))
                    return hi
                
                
       #%%         
#            beg_size = canvas_size
          
            # apply input noisy layer to real and fake samples
            
        hi = gaussian_noise_layer(hi, disc_noise_std)
            
        if not reuse:
            print('*** Discriminator summary ***')
            
            
        for block_idx, fmaps in enumerate(d_num_fmaps):
            
            hi = disc_block(block_idx, hi, 31, d_num_fmaps[block_idx],True, 'leakyrelu')
            
            if not reuse:
                print()
        
        if not reuse:
            print('discriminator deconved shape: ', hi.get_shape())
        
        hi_f = flatten(hi)  #keeps batch size, flatten everything else
        
        #hi_f = tf.nn.dropout(hi_f, self.keep_prob_var)
        
        d_logit_out = conv1d(hi, kwidth=1, num_kernels=1,
                             init=tf.truncated_normal_initializer(stddev=0.02),
                             name='logits_conv')
        
        d_logit_out = tf.squeeze(d_logit_out)  #removes dimensions of 1
        
        # all logits connected to 1 single neuron for binary classification
        d_logit_out = fully_connected(d_logit_out, 1, activation_fn=None)
        
        if not reuse:
            print('discriminator output shape: ', d_logit_out.get_shape())
            print('*****************************')
            
            
        return d_logit_out    
    
    
#%%

def vbn(self, tensor, name):
    if self.disable_vbn:
        class Dummy(object):
            # Do nothing here, no bnorm
            def __init__(self, tensor, ignored):
                self.reference_output=tensor
            def __call__(self, x):
                return x
        VBN_cls = Dummy
    else:
        VBN_cls = VBN
    if not hasattr(self, name):
        vbn = VBN_cls(tensor, name)
        setattr(self, name, vbn)
        return vbn.reference_output
    vbn = getattr(self, name)
    return vbn(tensor)





#%%

a=tf.random_normal([2500, 2**14])

b=generator(a, True)
















    
    
    
    