from __future__ import print_function
import tensorflow as tf
from ops import *
import numpy as np

#%%
def pre_emph(x, coeff=0.95):
    x0 = tf.reshape(x[0], [1,]) #OMG! Just make x[0] a 1-D tensor!!!
    diff = x[1:] - coeff * x[:-1]
    concat = tf.concat([x0, diff], 0)
    return concat

#%%
def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

#%%
def read_and_decode(filename_queue, canvas_size, preemph=0.):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'wav_raw': tf.FixedLenFeature([], tf.string),
                'noisy_raw': tf.FixedLenFeature([], tf.string),
            })
    wave = tf.decode_raw(features['wav_raw'], tf.int32)
    wave.set_shape(canvas_size) #reshaping to the desired shape
    
    # projecting to [-1,1]
    wave = (2./65535.) * tf.cast((wave - 32767), tf.float32) + 1.
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    
    noisy.set_shape(canvas_size)
    noisy = (2./65535.) * tf.cast((noisy - 32767), tf.float32) + 1.

    if preemph > 0:
        wave = tf.cast(pre_emph(wave, preemph), tf.float32)
        noisy = tf.cast(pre_emph(noisy, preemph), tf.float32)

    return wave, noisy


#%% TESt

path_name = "/vol/grid-solar/sgeusers/hsadeghi/segan_data/segan.tfrecords"

filename_queue = tf.train.string_input_producer([path_name])

wave, noise = read_and_decode(filename_queue, 2**14, 0.95)

sess=tf.Session()

wav1, noi1= sess.run([wave, noise])

#wav2, noi2= sess.run([wave, noise])


#print('diff', sess.run(tf.reduce_sum(tf.add(wav1,-wav2))))

print('shape wave', sess.run(tf.shape(wav1)[0]))

print('shape noise', sess.run(tf.shape(wav1)[1]))

    



















