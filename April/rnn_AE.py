'''
@author: hsadeghi
'''
#%%
from __future__ import division, print_function, absolute_import

import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io as si #for reading the data
#for stochastic Neurons
#import stochastic_neuron as sn

#import tflearn
#from rnn_layer import rnn_layer

#%%
#loading data
mat = si.loadmat('com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
# >> median(abs(concat_wav)) = 0.0376

n_data=data.shape[1];
#input_dim =int(fs/25); # Frames size of speech signal
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


n_batch = 5000;  #int(n_training/batch_size)

learning_rate = 0.01
training_epochs = 1
batch_size = 128

# Network Parameters

drop_out_p=0.8;


full_width=5;
rnn_width=4;
binary_width=12;
num_steps=2;


display_step = 100


#%% ##############################################################################
# input, weights and biases

X = tf.placeholder("float", [None, input_dim])
#init_state= tf.placeholder("float", [2, None, rnn_width])
n_time_samples= batch_size #tf.placeholder("int32", [None])



std_weight=( 2.0 / max( [full_width, rnn_width] ))**0.5;
std_bias=0.01;

def w_b_gen(shape, stddev_param):
    
    weight= tf.Variable(tf.random_normal(shape, mean=0.0, stddev=stddev_param)); 
    #bias= tf.Variable(tf.random_normal([out_dim],  mean=0.0, stddev=std_bias));
    
    return weight


#%% Generating weights

weights={}
biases={}

weights['enc_full']  = w_b_gen( [input_dim, full_width                ], std_weight)
weights['enc_rnn_1'] = w_b_gen( [4, full_width + rnn_width , rnn_width], std_weight)
weights['enc_rnn_2'] = w_b_gen( [4, rnn_width + rnn_width ,  rnn_width], std_weight)
weights['middle']    = w_b_gen( [rnn_width, binary_width]              , std_weight)

weights['dec_rnn_1'] = w_b_gen( [4,  binary_width + rnn_width , rnn_width], std_weight)
weights['dec_rnn_2'] = w_b_gen( [4, rnn_width + rnn_width , rnn_width],     std_weight)
weights['dec_full']  = w_b_gen( [rnn_width, full_width],                    std_weight)
weights['out']       = w_b_gen( [full_width, input_dim],                    std_weight)

 # weights is [ 4, in_dim + num_neurons , num_neurons]
 # biases is [4, num_neorons]
biases['enc_rnn_1'] = w_b_gen([4, rnn_width], std_bias)
biases['enc_rnn_2'] = w_b_gen([4, rnn_width], std_bias)
biases['dec_rnn_1'] = w_b_gen([4, rnn_width], std_bias)
biases['dec_rnn_2'] = w_b_gen([4, rnn_width], std_bias)


#%%
# Batch normalization
def batch_norm(x, W):
    
    num_neurons=W.get_shape()[1].value;
    epsilon=1e-5;
    z_BN = tf.matmul(x,W)
    batch_mean, batch_var = tf.nn.moments(z_BN,[0])
    scale = tf.Variable(tf.ones([num_neurons]))
    beta = tf.Variable(tf.zeros([num_neurons]))
    x_BN = tf.nn.batch_normalization(z_BN,batch_mean,batch_var,beta,scale,epsilon)

    return x_BN



#%% RNN layer%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def rnn_layer(x, previous_state, w, b):
     
     # x is [num_batches , input_dim]
     # weights is [ 4, in_dim + num_neurons , num_neurons]
     # biases is [4, num_neorons]
     # previous_state is [2 , num_neurons] includes both h & c
       
     h_past= previous_state[0]
     c_past= previous_state[1]
 

#      n_time_samples=  x.get_shape()[0].value   # equals batch_size in the training phase
#     n_time_samples = batch_size
     
     
#     h_past=tf.reshape( tf.tile (h_past,[n_time_samples]), [n_time_samples, -1])
#     c_past=tf.reshape( tf.tile (c_past,[n_time_samples]), [n_time_samples, -1])
#     
     
     Th_i = tf.matmul( tf.concat([x, h_past ], 1), w[0])
     Th_f = tf.matmul( tf.concat([x, h_past ], 1), w[1])
     Th_o = tf.matmul( tf.concat([x, h_past ], 1), w[2])
     Th_g = tf.matmul( tf.concat([x, h_past ], 1), w[3])
     

     
     b_i=tf.reshape( tf.tile (b[0,:],[n_time_samples]), [n_time_samples, -1]) 
     b_o=tf.reshape( tf.tile (b[1,:],[n_time_samples]), [n_time_samples, -1])
     b_f=tf.reshape( tf.tile (b[2,:],[n_time_samples]), [n_time_samples, -1])
     b_g=tf.reshape( tf.tile (b[3,:],[n_time_samples]), [n_time_samples, -1])

     
     i = tf.add( b_i, tf.sigmoid( Th_i )) # These will be all column vectors
     o = tf.add( b_o, tf.sigmoid( Th_f ))
     f = tf.add( b_f, tf.sigmoid( Th_o ))
     g = tf.add( b_g, tf.tanh(    Th_g ))
     
     c= tf.add ( tf.multiply(f, c_past) , tf.multiply(i, g) )
     
     h= tf.multiply ( o, tf.tanh(c) )
     
     state = tf.stack([h,c])
         
     output=h
     
     return output, state
 
#%%##############################################################################
#%% Full layer

def full_layer (x, w):
    
    layer=batch_norm(x, w)
    layer=tf.nn.tanh(layer);  
    
    return layer
        
#%%##############################################################################
# Building the encoder

def encoder(x, enc_state_1, enc_state_2, weights, biases):
    
    enc_full = full_layer (x, weights['enc_full'])
    
    # enc_full=batch_norm(enc_full, weights['enc_rnn_1'])
            
    rnn_1 , enc_state_1 = rnn_layer(enc_full, enc_state_1, weights['enc_rnn_1'], biases['enc_rnn_1'] )
    
    # rnn_1=batch_norm(rnn_1, weights['enc_rnn_2'])
    
    rnn_2 , enc_state_2 = rnn_layer(rnn_1, enc_state_2, weights['enc_rnn_2'], biases['enc_rnn_2'])
    
    full_middle= full_layer (rnn_2, weights['middle'])

    
    return full_middle, enc_state_1, enc_state_2


#%%    
def decoder(x, dec_state_1, dec_state_2, weights, biases):
    
    # x=batch_norm(x, weights['dec_rnn_1'])

    rnn_1 , dec_state_1 = rnn_layer(x, dec_state_1, weights['dec_rnn_1'], biases['dec_rnn_1'])
    
    # rnn_1=batch_norm(rnn_1, weights['dec_rnn_2'])
    
    rnn_2 , dec_state_2 = rnn_layer(rnn_1, dec_state_2, weights['dec_rnn_2'], biases['dec_rnn_2'])
    
    dec_full = full_layer (rnn_2, weights['dec_full'])
    
    full_out = full_layer (dec_full, weights['out'])
    
    
    return full_out, dec_state_1, dec_state_2

###############################################################s
# Construct the residual model

state_enc_1 = tf.zeros([2, n_time_samples, rnn_width])
state_enc_2 = state_dec_1 = state_dec_2 = state_enc_1

residue=X # '- zero output' in fact


for _ in range(num_steps):
    
#    print('iteration', _)
    
    encoder_output, state_enc_1, state_enc_2 = encoder(residue, state_enc_1, state_enc_2, weights, biases)
    
    decoder_output, state_dec_1, state_dec_2 = decoder(encoder_output, state_dec_1, state_dec_2, weights, biases)
    
    residue= X - decoder_output
    



#%% Cost and optimization setup

y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow( y_true - decoder_output , 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate,  epsilon=1e-8).minimize(cost)


#%%##############################################################################
# Training
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph, mean=0.0, stddev=1))

#with tf.Session() as sess:
sess=tf.Session();
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

sess.run(init)


# Training cycle
cost_vector=[];

for epoch in range(training_epochs):
    # Loop over all batches
#        small_cost_occurance=0;
    
    for  i in range(n_batch):
        start_ind = i #np.random.randint(0, n_training-batch_size );  # For shuffling
        batch_xs= training_data[ start_ind : start_ind + batch_size, :];
        
        
        
#            batch_xs=batch_xs.reshape(n_input, batch_size);
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, n_time_samples:batch_size})

    # Display logs per epoch step
        if i % display_step == 0:
            print("Epoch:", '%02d' % (epoch+1),
                  "i:", '%04d' % (i+1),
                  "cost=", "{:.9f}".format(c))   
            
            cost_vector+=[c]

print("Optimization Finished!")

#%%##########################################################################
# Testing the network performance
training_error=sess.run(cost, feed_dict={X: training_data,
                                         n_time_samples:n_training})**0.5

y_pred_test, y_true_test, test_error = sess.run([decoder_output, y_true, cost],
                                                feed_dict={X: test_data,
                                                           n_time_samples:n_test})

test_error=test_error**0.5

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


#print('architecture ', hid_size)
print('learning_rate= ', learning_rate)
print('num_quantization_steps= ', num_steps)

# Plotting results
#plt.plot(cost_vector)

#%%##########################################################################
# Savings network

AE_output={};
AE_output['y_pred_test']=y_pred_test;
AE_output['y_true_test']=y_true_test;

si.savemat("/home/hsadeghi/Dropbox/research codes/April/AE_output.mat",
           AE_output);
          
           
sess.close()
