"""
Created on Tue Mar  7 15:39:57 2017
#@author: hsadeghi
code copied from 
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
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
input_dim = int(fs/25); # Frames size of speech signal


n_data=n_data - n_data % input_dim; # make length divisible by input_dim
data=data[0,0:n_data]; # clipping teh rest
# Reshaping data
data=data.reshape([int(n_data/input_dim), input_dim])
n_data=data.shape[0];


training_percentage=70;
n_training= int(np.floor(training_percentage/100.*n_data));
training_data=data[ 0:n_training , : ];

test_data=data[ n_training:n_data, : ];
n_test=test_data.shape[0];


#%% Parameters

learning_rate = 0.001
training_epochs = 1
batch_size = 10

# Network Parameters

#hid_size=[512,512,512];
hid_size=[512, 512, 512];
n_hidden_binary=64;
num_quantization_steps=8;

n_hid=len(hid_size);
display_step = 100

#%% ##############################################################################
# tf Graph input

X = tf.placeholder("float", [None, input_dim])

std_init_W=0.1;
std_init_bias=0.01;

weights={};
biases={};

weights['encoder_h'+str(1)]= tf.Variable(tf.random_normal([input_dim, hid_size[0]], mean=0.0, stddev=std_init_W));                                
biases['encoder_b'+str(1)]= tf.Variable(tf.random_normal([hid_size[0]],  mean=0.0, stddev=std_init_bias));
weights['decoder_h'+str(1)]= tf.Variable(tf.random_normal([n_hidden_binary, hid_size[-1]], mean=0.0, stddev=std_init_W));
biases['decoder_b'+str(1)]=tf.Variable(tf.random_normal([hid_size[-1]], mean=0.0, stddev=std_init_bias));

for i in range(1,n_hid):

    weights['encoder_h'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[i-1], hid_size[i]], mean=0.0, stddev=std_init_W));                                
    biases['encoder_b'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[i]],  mean=0.0, stddev=std_init_bias));
    
    weights['decoder_h'+str(i+1)]= tf.Variable(tf.random_normal([hid_size[-i], hid_size[-i-1]], mean=0.0, stddev=std_init_W));
    biases['decoder_b'+str(i+1)]=tf.Variable(tf.random_normal([hid_size[-i-1]], mean=0.0, stddev=std_init_bias));

## Middle one used for quantization (outside of the loop)
weights['middle']=tf.Variable(tf.random_normal( [hid_size[n_hid-1], n_hidden_binary], mean=0.0, stddev=std_init_W));
biases['middle']= tf.Variable(tf.random_normal( [n_hidden_binary],  mean=0.0, stddev=std_init_bias));
# Last oen used to reconstruct input (i.e. AE output)
weights['last']=tf.Variable(tf.random_normal( [hid_size[0], input_dim], mean=0.0, stddev=std_init_W));
biases['last']= tf.Variable(tf.random_normal( [input_dim],  mean=0.0, stddev=std_init_bias));



#%%##############################################################################
# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer={};
    layer[str(1)] = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h'+str(1)]), biases['encoder_b'+str(1)]))
    for i in range(2, n_hid+1):
        layer[str(i)] = tf.nn.tanh(tf.add(tf.matmul(layer[str(i-1)], weights['encoder_h'+str(i)]), biases['encoder_b'+str(i)]))
    #  Quantizers    
    layer_middle = tf.nn.tanh( tf.add( tf.matmul(layer[str(n_hid)], weights['middle']), biases['middle']))    
    return layer_middle


# Building the decoder
def decoder(x):
    layer={};
    layer[str(1)] = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    
    for i in range(2,n_hid+1):
        layer[str(i)] = tf.nn.tanh(tf.add(tf.matmul(layer[str(i-1)], weights['decoder_h'+str(i)]), biases['decoder_b'+str(i)]))
        
    layer_last=tf.nn.tanh(tf.add(tf.matmul(layer[str(n_hid)], weights['last']), biases['last']))
    return layer_last
    
#%%#############################################################################
# Construct teh residual model

encoder_op={};
decoder_op={};
residue={};

# fisrt step
encoder_op[str(1)] = encoder(X);
decoder_op[str(1)] = decoder(encoder_op[str(1)])

y_pred=decoder_op[str(1)];
residue[str(1)]=tf.subtract(X, decoder_op[str(1)]);

# AEing the residue
for i in range(2, num_quantization_steps+1):
    
    encoder_op[str(i)] = encoder(residue[str(i-1)])
    decoder_op[str(i)] = decoder(encoder_op[str(i)])
    
    y_pred=tf.add( y_pred , decoder_op[str(i)] )
    residue[str(i)]=tf.subtract(residue[str(i-1)], decoder_op[str(i)]);

# Ground truth
y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow(y_true - y_pred, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-04).minimize(cost)


#%%##############################################################################
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph, mean=0.0, stddev=1))

#with tf.Session() as sess:
sess=tf.Session();

sess.run(init)


n_batch = int(n_training/batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
#        small_cost_occurance=0;
    
    for  i in range(n_batch):
        start_ind=np.random.randint(0, n_training-batch_size );  # For shuffling
        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
#            batch_xs=batch_xs.reshape(n_input, batch_size);
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))    

print("Optimization Finished!")

#%%##########################################################################
# Testing the network performance

training_error=sess.run(cost, feed_dict={X: training_data})**0.5

y_pred_test, y_true_test, test_error = sess.run([y_pred, y_true, cost], feed_dict={X: test_data})

test_error=test_error**0.5

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


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


