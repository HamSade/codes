#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 18:04:15 2017

@author: hsadeghi
"""
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data

# for stochastic Neurons
#import stochastic_neuron as sn


#%%
#loading data
mat = si.loadmat('/home/hsadeghi/Downloads/PDA/my_data/com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
# >> median(abs(concat_wav)) = 0.0376

n_data=data.shape[1];
#input_dim =int(fs/25); # Frames size of speech signal
input_dim =100;

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




#%% Global config variables

learning_rate = 0.1

full_width=512;

hidden_width = 512
number_of_layers=2

n_hidden_binary=16;

num_steps = 2 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 100


n_batch = 2000; # int(n_training/batch_size)


training_epochs = 1
display_step = 100

#%% ##############################################################################
# tf Graph input

X = tf.placeholder("float", [None, input_dim])
previous_state_enc=tf.placeholder("float", [1, hidden_width])
previous_state_dec=tf.placeholder("float", [1, hidden_width])


std_init_W=0.1;
std_init_bias=0.01;

weights={};
biases={};

weights['encoder_h'+str(1)]= tf.Variable(tf.random_normal([input_dim, full_width], mean=0.0, stddev=std_init_W));                                
biases['encoder_b'+str(1)]= tf.Variable(tf.random_normal([full_width],  mean=0.0, stddev=std_init_bias));

weights['middle']=tf.Variable(tf.random_normal( [hidden_width, n_hidden_binary], mean=0.0, stddev=std_init_W));
biases['middle']= tf.Variable(tf.random_normal( [n_hidden_binary],  mean=0.0, stddev=std_init_bias));

weights['last']=tf.Variable(tf.random_normal( [hidden_width, full_width], mean=0.0, stddev=std_init_W));
biases['last']= tf.Variable(tf.random_normal( [full_width],  mean=0.0, stddev=std_init_bias));

weights['output']= tf.Variable(tf.random_normal([full_width, input_dim], mean=0.0, stddev=std_init_W));
biases['output']=tf.Variable(tf.random_normal([input_dim], mean=0.0, stddev=std_init_bias));


#%%
# Batch normalization


def BatchNormalization(x, W, num_neurons):
    epsilon=1e-3;
    z_BN = tf.matmul(x,W)
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
    scale = tf.Variable(tf.ones([num_neurons]))
    beta = tf.Variable(tf.zeros([num_neurons]))
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN


#%% RNN layer definition

#def rnn_layer_gen(state_size):
#       layer= tf.contrib.rnn.BasicRNNCell (state_size,\
#                                           input_size=input_dim,\
#                                           activation=tf.tanh)
#       return layer


with tf.variable_scope('rnn_cell'):
    W = tf.get_variable('W', [num_classes + state_size, state_size])
    b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

def rnn_cell(rnn_input, state):
    with tf.variable_scope('rnn_cell', reuse=True):
        W = tf.get_variable('W', [num_classes + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
    return tf.tanh(tf.matmul(tf.concat([rnn_input, state], 1), W) + b)
#%%##############################################################################
# Building the encoder

def encoder(x, previous_state):
       
       #    layer[str(1)] = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h'+str(1)]), biases['encoder_b'+str(1)]))
       layer_1 = BatchNormalization(x, weights['encoder_h'+str(1)],hidden_width);
       layer_1 = tf.nn.tanh(layer_1);
       
       
       rnn_layer=rnn_layer_gen(hidden_width)
       rnn_net=tf.contrib.rnn.MultiRNNCell( [rnn_layer] * number_of_layers,\
                                            state_is_tuple=True)
#       initial_state=state = tf.zeros([batch_size, rnn_net.state_size])
       
       # Critical part where we get the state of rnn
       output_rnn, state_rnn = rnn_net(layer_1, previous_state)

             
       #layer_middle=tf.nn.tanh(tf.add(tf.matmul(layer[str(n_hid)],weights['middle']), biases['middle']));
       layer_middle=BatchNormalization(output_rnn, weights['middle'], n_hidden_binary)
       layer_middle=tf.nn.tanh(layer_middle)

       
       return layer_middle, state_rnn

#%% Building the decoder
def decoder(x, previous_state):
       
       
       rnn_layer=rnn_layer_gen(hidden_width)
       
       rnn_net=tf.contrib.rnn.MultiRNNCell( [rnn_layer] * number_of_layers,\
                                            state_is_tuple=True)
       
       
       # Critical part where we get the state of rnn
       output_rnn, state_rnn = rnn_net(x, previous_state)


       layer_last=BatchNormalization(output_rnn, weights['last'], full_width)
       layer_last=tf.nn.tanh(layer_last) 
             
       
       layer_output=BatchNormalization(layer_last, weights['output'], input_dim)
       layer_output=tf.nn.tanh(layer_output) 
    
       return layer_last, state_rnn
    

#%%#############################################################################
# Construct teh residual model

encoder_op={};
decoder_op={};
residue={};


# fisrt step

#previous_state_enc =tf.zeros([batch_size, hidden_width])
encoder_op, state_enc  = encoder(X, previous_state_enc)
decoder_op, state_dec = decoder(encoder_op,previous_state_dec) 

y_pred, _=decoder_op;
residue=tf.subtract(X, y_pred);

# Ground truth
y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow(y_true - y_pred, 2))

#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


#%%##############################################################################
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph, mean=0.0, stddev=1))

#with tf.Session() as sess:
#sess=tf.Session();
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


sess.run(init)


# Training cycle
cost_vector=[];

for epoch in range(training_epochs):
    
       
       for  mini_batch in range(n_batch):
              
              start_ind=np.random.randint(0, n_training-batch_size*num_steps ,1)  
              
              
              batch_xs= training_data[ start_ind : start_ind + batch_size*num_steps, :];                       
       
              state_enc_tmp = state_dec_tmp = tf.zeros([batch_size, hidden_width])
              residue_tmp = tf.zeros([ batch_size, input_dim])
              
              for step in range(num_steps):
                     
                     net_input=batch_xs[ step * batch_size : (step+1) * batch_size, : ]
                     net_input=net_input-residue_tmp;
                                          
                     residue_tmp, state_enc_tmp, state_dec_tmp=sess.run([residue, state_enc, state_dec],
                                                          feed_dict={X: net_input,                                                                    
                                                               previous_state_enc: state_enc_tmp,
                                                               previous_state_dec: state_dec_tmp })
                          
 
              c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs,\
                              previous_state_enc: state_enc_tmp,\
                              previous_state_dec: state_dec_tmp })
              
              # Display logs per epoch step
              if mini_batch % display_step == 0:
                     print("Epoch:", '%02d' % (epoch+1),
                           "i:", '%04d' % (mini_batch+1),
                           "cost=", "{:.9f}".format(c))   
            
              cost_vector+=[c]

print("Optimization Finished!")

#%%##########################################################################
# Testing the network performance

training_error=sess.run(cost, feed_dict={X: training_data})**0.5

y_pred_test, y_true_test, test_error = sess.run([y_pred, y_true, cost], feed_dict={X: test_data})

test_error=test_error**0.5

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


print('architecture ', [[hidden_width]*number_of_layers, n_hidden_binary, [hidden_width]*number_of_layers ])
print('learning_rate= ', learning_rate)
print('n_hidden_binary= ', n_hidden_binary)
print('num_steps= ', num_steps)

# Plotting results
#plt.plot(cost_vector)

#%%##########################################################################
# Savings network
#saver = tf.train.Saver()
#save_path = saver.save(sess, "/home/hsadeghi/Dropbox/research codes/",
#                       "binary_full_2step_4bit_AE.ckpt")

#    print("Model saved in file: %s" % save_path)
AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/research codes/AE_output.mat",
           AE_output);


#%% Building graph

#writer = tf.summary.FileWriter('/Dropbox/research codes/log', sess.graph)
