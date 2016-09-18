import tensorflow as tf
import time
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers
import cPickle

def unpickle(file):

    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def conv2d(inputs, output_size, kernel_size, stride, 
			weights_initializer=tf.contrib.layers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer,
			activation_fn=tf.nn.relu, trainable=True, padding='VALID', name='conv2d'):
	
	kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[1], output_size]
	stride  = [1, 1, stride[0], stride[1]]
	with tf.variable_scope(name):
		w = tf.get_variable('w', kernel_shape,
			tf.float32, initializer=weights_initializer, trainable=trainable)
		conv = tf.nn.conv2d(inputs, w, stride, padding, data_format='NCHW')
		b = tf.get_variable('b', [output_size], tf.float32, initializer=biases_initializer, trainable=trainable)
		out = tf.nn.bias_add(conv, b, 'NCHW')
		
	if activation_fn != None:
		out = activation_fn(out)
	return out, w, b

def linear(inputs, output_size,	
            weights_initializer=initializers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer,
			activation_fn=None, batch_norm=True, name='linear'):
	
	shape = inputs.get_shape().as_list()
	with tf.variable_scope(name):
		w = tf.get_variable('w', [shape[1], output_size], tf.float32,
						initializer=weights_initializer)
		b = tf.get_variable('b', [output_size],
						initializer=biases_initializer)
		out = tf.nn.bias_add(tf.matmul(inputs, w), b)
	if batch_norm:
		out = tf.contrib.layers.batch_norm(out)

	if activation_fn is None:
		return out, w, b
	else:
		return activation_fn(out), w, b

def get_time():
	return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())

