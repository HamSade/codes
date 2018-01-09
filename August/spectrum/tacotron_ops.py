#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:18:53 2017

@author: hsadeghi

By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/tacotron
"""

from __future__ import print_function

from hyperparams import Hyperparams as hp
import tensorflow as tf


def embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    '''Embeds a given tensor. 
    
    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                      lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)   
 
def normalize(inputs, 
              type="bn",
              decay=.999,
              epsilon=1e-8,
              is_training=True, 
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but 
        the last dimension. Or if type is `ln`, the normalization is over 
        the last dimension. Note that this is different from the native 
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py` 
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type=="bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        
        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank==2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank==3:
                inputs = tf.expand_dims(inputs, axis=1)
            
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               zero_debias_moving_mean=True,
                                               fused=True,
                                               reuse=reuse)
            # restore original shape
            if inputs_rank==2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank==3:
                outputs = tf.squeeze(outputs, axis=1)
        else: # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs, 
                                               decay=decay,
                                               center=True, 
                                               scale=True, 
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)    
    elif type in ("ln",  "ins"):
        reduction_axis = -1 if type=="ln" else 1   
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
        
            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
    else:
        outputs = inputs
    
    if activation_fn:
        outputs = activation_fn(outputs)
                
    return outputs

def conv1d(inputs, 
           filters=None, 
           size=1, 
           rate=1, 
           padding="SAME", 
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        
        if filters is None:
            filters = inputs.get_shape().as_list[-1]
        
        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn, 
                "use_bias":use_bias, "reuse":reuse}
        
        outputs = tf.layers.conv1d(**params)
    return outputs

def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is, 
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
    
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, hp.embed_size//2, 1) # k=1
        for k in range(2, K+1): # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, hp.embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = normalize(outputs, type=hp.norm_type, is_training=is_training, 
                            activation_fn=tf.nn.relu)
    return outputs # (N, T, Hp.embed_size//2*K)

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results 
        are concatenated.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
            
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, 
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, 
                                           dtype=tf.float32)
            return outputs

def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    '''Applies a GRU to `inputs`, while attending `memory`.
    Args:
      inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
      memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
      seqlens: A 1d tensor with shape of [N,], dtype of int32.
      num_units: An int. Attention size.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A 3d tensor with shape of [N, T, num_units].    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, 
                                                                   memory, 
                                                                   normalize=True,
                                                                   probability_fn=tf.nn.softmax)
        decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, num_units)
        outputs, _ = tf.nn.dynamic_rnn(cell_with_attention, inputs, 
                                       dtype=tf.float32) #( N, T', 16)
    return outputs

def prenet(inputs, is_training=True, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder.
    Args:
      inputs: A 3D tensor of shape [N, T, hp.embed_size].
      is_training: A boolean.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=hp.embed_size, activation=tf.nn.relu, name="dense1")
        outputs = tf.nn.dropout(outputs, keep_prob=.5 if is_training==True else 1., name="dropout1")
        outputs = tf.layers.dense(outputs, units=hp.embed_size//2, activation=tf.nn.relu, name="dense2")
        outputs = tf.nn.dropout(outputs, keep_prob=.5 if is_training==True else 1., name="dropout2") 
    return outputs # (N, T, num_units/2)

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387
    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C
    return outputs
Contact GitHub API Training Shop Blog About
© 2017 GitHub, Inc. Terms Privacy Security Status Help