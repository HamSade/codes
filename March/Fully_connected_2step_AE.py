'''code copied from 
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
'''

from __future__ import division, print_function, absolute_import



import tensorflow as tf
#import data_preprocessing as dp;
import numpy as np
#import matplotlib.pyplot as plt
import scipy.io #for reading teh data


#loading data
mat = scipy.io.loadmat('/home/hsadeghi/Downloads/PDA/my_data/com_concat_signal.mat')   ;
fs=np.asscalar(mat['fs_new']);

data=mat['concat_wav'];  # 100 files of 5 summed speakers, each file a few secs
data=np.array(data); 
#data.shape: (1, 15603279)
n_data=data.shape[1];
input_dim = int(fs/25); # Frames size of speech signal


n_data=n_data - n_data % input_dim; # make length divisible by input_dim
data=data[:,0:n_data];
# Reshaping data
data=data.reshape([int(n_data/input_dim), input_dim])
n_data=data.shape[0];


training_percentage=70;
n_training= int(np.floor(training_percentage/100.*n_data));
training_data=data[ 0:n_training , : ];

test_data=data[ n_training:n_data, : ];
n_test=test_data.shape[0];

##########################################################
##########################################################
##########################################################

# Parameters
learning_rate = 0.1
training_epochs = 5
batch_size = 10

display_step = 100
examples_to_show = 100


# Network Parameters
n_hidden_1 = 512 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 512;
n_hidden_binary=4;

###############################################################################
# tf Graph input

X = tf.placeholder("float", [None, input_dim])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([input_dim, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_binary])),
    
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_binary, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, input_dim])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_binary])),
    
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([input_dim])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   biases['encoder_b4'])) 
    return layer_4


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4

# Construct model
encoder_op_1 = encoder(X)
decoder_op_1 = decoder(encoder_op_1)

# Autoencoding the residual
res_1=tf.subtract(X, decoder_op_1);
encoder_op_2=encoder(res_1)
decoder_op_2=decoder(encoder_op_2)


# Prediction
y_pred = tf.add(decoder_op_2, decoder_op_1);
y_true = X; # Targets (Labels) are the input data. Cause it's an Autoencoder!!


# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean( tf.pow(y_true - y_pred, 2) )
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


###############################################################################
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph

#with tf.Session() as sess:
sess=tf.Session();

sess.run(init)

small_cost_occurance=0;

n_batch = int(n_training/batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
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
            
#            if c < 1e-3:
#                
#                small_cost_occurance+=1;
#                if small_cost_occurance>10:
#                    break
#                
#            else:
#                continue
#            break
        

print("Optimization Finished!")

###########################################################################
# Testing the network performance

training_error=sess.run(cost, feed_dict={X: training_data})
test_error = sess.run(cost, feed_dict={X: test_data})

#_, test_error = sess.run([optimizer, cost], feed_dict={X: test_data})
print( 'training_error', "{:.9f}".format(training_error))
print( 'test_error', "{:.9f}".format(test_error))


# Saving results
saver = tf.train.Saver()
save_path = saver.save(sess, "/home/hsadeghi/Dropbox/research codes/full_2step_4bit_AE.ckpt")
print("Model saved in file: %s" % save_path)

#sess.close()

