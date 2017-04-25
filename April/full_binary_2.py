#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:32:21 2017

@author: hsadeghi


This is a binary version with bits at the ouput. Each bit is connected to input and all previous bits
"""

#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data

#for stochastic Neurons
from binary_quantizer import binary_quantizer

from binary import bitwise_layer


#%%
#loading data
mat = si.loadmat('com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
# >> median(abs(concat_wav)) = 0.0376

n_data=data.shape[1];
input_dim =128;

n_data=n_data - n_data % input_dim; # make length divisible by input_dim
data=data[0,0:n_data]; # clipping teh rest
# Reshaping data
data=data.reshape([int(n_data/input_dim), input_dim])
n_data=data.shape[0];


training_percentage=90;
n_training= int(np.floor(training_percentage/100.*n_data));
training_data=data[ 0:n_training , : ];

test_data=data[ n_training:n_data, : ];
n_test=test_data.shape[0];


#%% Parameters

learning_rate = 0.01
training_epochs = 1
batch_size = 128
num_output_bits=16;


# Network Parameters
hid_size=[10, 10, 10];
n_hidden_binary=5;
num_quantization_steps=2;

n_hid=len(hid_size);


n_batch = int(2 * n_training / batch_size)
display_step = 100


#%% ##############################################################################
# tf Graph input

X = tf.placeholder("float", [None, input_dim])
mode = tf.placeholder("float", None)  # for quantizing neurons


std_init_W=(2.0/max(hid_size))**0.5;
std_init_bias=0.01;
    

weights={};
biases={};

weights['encoder_h'+str(1)] = tf.Variable(tf.random_normal([input_dim, hid_size[0]], mean=0.0, stddev=std_init_W));                                
biases ['encoder_b'+str(1)] = tf.Variable(tf.random_normal([hid_size[0]],  mean=0.0, stddev=std_init_bias));

weights['decoder_h'+str(1)]= tf.Variable(tf.random_normal([n_hidden_binary, hid_size[-1]], mean=0.0, stddev=std_init_W));
biases['decoder_b'+str(1)]=tf.Variable(tf.random_normal([hid_size[-1]], mean=0.0, stddev=std_init_bias));

for i in range(1,n_hid):

    weights['encoder_h'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[i-1], hid_size[i]], mean=0.0, stddev=std_init_W));                                
    biases['encoder_b'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[i]],  mean=0.0, stddev=std_init_bias));
    
    weights['decoder_h'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[-i], hid_size[-i-1]], mean=0.0, stddev=std_init_W));
    biases['decoder_b'+str(i+1)]=tf.Variable(tf.random_normal([hid_size[-i-1]], mean=0.0, stddev=std_init_bias));

weights['middle']=tf.Variable(tf.random_normal( [hid_size[n_hid-1], n_hidden_binary], mean=0.0, stddev=std_init_W));
biases['middle']= tf.Variable(tf.random_normal( [n_hidden_binary],  mean=0.0, stddev=std_init_bias));

weights['last']=tf.Variable(tf.random_normal( [hid_size[0], input_dim], mean=0.0, stddev=std_init_W));
biases['last']= tf.Variable(tf.random_normal( [input_dim],  mean=0.0, stddev=std_init_bias));



# binarizer layer paramaters
#weights['binary'] = tf.Variable(tf.random_normal( [num_output_bits+1, num_output_bits], mean=0.0, stddev=std_init_W));

#wb = []
#for i in range(num_output_bits):
#    wb.append(tf.random_normal([i+1,1], stddev=std_init_W))

wb = [tf.Variable(tf.random_normal([1,1], stddev=std_init_W ))]
for i in range(1, num_output_bits):
    wb.append(tf.Variable(tf.random_normal([ i + 1 , 1], stddev=std_init_W ) ) )


weights['binary'] = wb; #tf.Variable(wb);
biases['binary']= tf.Variable(tf.random_normal( [num_output_bits], stddev=std_init_bias));

#%%

# Batch normalization with shared scale and beta variables
#def BatchNormalization(x, W, b_scope):
#    
#    num_neurons=W.get_shape()[1].value;
#    epsilon=1e-3;
#    z_BN = tf.matmul(x,W)
#    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
#    
#    try:
#        with tf.variable_scope(b_scope):
#        
#            scale = tf.get_variable(tf.ones([num_neurons]))
#            beta = tf.get_Variable(tf.zeros([num_neurons]))
#    
#    except ValueError:
#        with tf.variable_scope(b_scope, reuse=True):
#            
##            tf.get_variable_scope().reuse_variables()
#            scale = tf.get_variable(tf.ones([num_neurons]))
#            beta = tf.get_Variable(tf.zeros([num_neurons]))
#    
#        x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)
#
#    return x_BN

#%%
def BatchNormalization(x, W):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
    scale = tf.Variable(tf.ones([num_neurons]))
    beta = tf.Variable(tf.zeros([num_neurons]))
    
#    scale = tf.ones([num_neurons])
#    beta = tf.zeros([num_neurons])

    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN

#%%##############################################################################
# Define quantizer

b_q=binary_quantizer(tf)


#%%
def encoder(x):

    layer={};
    layer[str(1)]=BatchNormalization(x, weights['encoder_h'+str(1)])
    layer[str(1)]=tf.nn.tanh(layer[str(1)]);  
        
    for i in range(2, n_hid+1):
        layer[str(i)] = BatchNormalization(layer[str(i-1)], weights['encoder_h'+str(i)]);

        layer[str(i)]= tf.nn.tanh(layer[str(i)]);
       
        
    # Quantized version    
    layer_middle=BatchNormalization(layer[str(n_hid)], weights['middle'])
    layer_middle=tf.nn.tanh(layer_middle)
    layer_middle = b_q(layer_middle, mode)  # For quantization
    
    
    
    return layer_middle


#%%
def decoder(x):

    layer={};
    layer[str(1)] = BatchNormalization(x, weights['decoder_h1']);
    layer[str(1)]=tf.nn.tanh(layer[str(1)]);
    
    for i in range(2,n_hid+1):
        layer[str(i)] = BatchNormalization(layer[str(i-1)], weights['decoder_h'+str(i)])
        layer[str(i)]= tf.nn.tanh(layer[str(i)])
 
    layer_last=BatchNormalization(layer[str(n_hid)], weights['last'])
    layer_last=tf.add( layer_last , biases['last'] ); 
    
    # Added for binarizing output (might convert tanh to linear if fully continuous)
    layer_last = tf.tanh (layer_last)
    
    return layer_last
    
#%%#############################################################################
# Construct teh residual model

encoder_op={};
decoder_op={};
residue={};
y_pred={};

# fisrt step
encoder_op[str(1)] = encoder(X);
decoder_op[str(1)] = decoder(encoder_op[str(1)])

y_pred[str(1)]=decoder_op[str(1)];
residue[str(1)]=tf.subtract(X, decoder_op[str(1)]);

# AEing the residue
for i in range(2, num_quantization_steps+1):
    
    encoder_op[str(i)] = encoder(residue[str(i-1)])
    decoder_op[str(i)] = decoder(encoder_op[str(i)])
    
    y_pred[str(i)]=tf.add( y_pred[str(i-1)] , decoder_op[str(i)] )
    residue[str(i)]=tf.subtract(residue[str(i-1)], decoder_op[str(i)]);

#%% Cost calculation

# continuous version
y_true=X
y_pred_final = y_pred[str(num_quantization_steps)];


######### Building binarizer

binarizer_vec = []

for i in range(num_output_bits-1):
    
    binarizer_vec.append( [2.** (-i-1)] )

##############################
# Binarizing the outputs
#out_pred = []
out_pred_final = []

for i in range(input_dim):
   
#    print('iteration', i)
    binary_rep = bitwise_layer( tf.reshape(y_pred_final[: , i], [-1 , 1]),  num_output_bits, weights['binary'], biases['binary'], tf, mode)
    
    float_rep = tf.multiply ( tf.reshape( binary_rep [:,0], [-1,1]) , tf.matmul( 0.5 * (1. + binary_rep[:,1:]),
                                                                         binarizer_vec) )
     
    out_pred_final.append(float_rep)
#    out_pred_final = tf.concat( [ out_pred_final, float_rep], axis=1)
    
#    out_pred = tf.concat( [ out_pred, binary_rep], axis=1)
#y_pred_binary = out_pred

y_pred_final = tf.stack(out_pred_final,1)

#%% Cost and optimization

cost = tf.reduce_mean( tf.pow(y_true - y_pred_final, 2))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


#%%##############################################################################
# Initializing the variables
init = tf.global_variables_initializer()
sess=tf.Session();
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
sess.run(init)

# Training cycle
cost_vector=[];

for epoch in range(training_epochs):
    
    for  i in range(n_batch):
        
#        start_ind=np.random.randint(0, n_training-batch_size );  # For shuffling
#        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
        
        start_ind=i;
        batch_xs= training_data[ start_ind* int(0.5*batch_size) :\
                                start_ind* int(0.5*batch_size) + batch_size, :];


        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, mode:0.0})
    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%02d' % (epoch+1),
                  "i:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(c))   
            
            cost_vector+=[c]

print("Optimization Finished!")

#%%##########################################################################
# Testing the network performance

training_error=sess.run(cost, feed_dict={X: training_data, mode:1.0})**0.5

y_pred_test, y_true_test, test_error = sess.run([y_pred_final, y_true, cost],
                                                        feed_dict={X: test_data, mode:1.0})

test_error=test_error**0.5

print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


print('architecture ', hid_size)
print('learning_rate= ', learning_rate)
print('n_hidden_binary= ', n_hidden_binary)
print('num_quantization_steps= ', num_quantization_steps)

# Plotting results
#plt.plot(cost_vector)

#%%##########################################################################
# Savings network

AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/April/AE_output.mat",
           AE_output);

sess.close()

#%% Building graph
#writer = tf.summary.FileWriter('/Dropbox/research codes/log', sess.graph)
