import tensorflow as tf
import time
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers
import cPickle
import pdb

def unpickle(file):

    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def pooling(inputs, kernel_size=None, stride=None, type='max', padding='SAME', name='pooling'):
	
	if kernel_size != None:
		kernel = [1, kernel_size[0], kernel_size[1], 1]
	if stride != None:
		stride = [1, stride[0], stride[1], 1]
	with tf.variable_scope(name):
		if type == 'max':
			out = tf.nn.max_pool(inputs, kernel, stride, padding=padding)
		elif type == 'average':
			out = tf.nn.avg_pool(inputs, kernel, stride, padding=padding)
		elif type == 'global_avg':
			input_shape = inputs.get_shape()
			out = tf.nn.avg_pool(inputs, [1,input_shape[1],input_shape[2],1], [1,input_shape[1],input_shape[2],1], padding='VALID')

	return out

def conv2d(inputs, output_size, kernel_size, stride, 
			weights_initializer=tf.contrib.layers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer, synthetic=False,
			batch_norm = True, conditional=False, label=None, 
			activation_fn=tf.nn.relu, padding='SAME', name='conv2d'):
	
	var = {}
	kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape()[-1], output_size]
	stride  = [1, 1, stride[0], stride[1]]
	with tf.variable_scope(name):
		var['w'] = tf.get_variable('w', kernel_shape,
			tf.float32, initializer=weights_initializer)
		conv = tf.nn.conv2d(inputs, var['w'], stride, padding=padding)
		var['b'] = tf.get_variable('b', [output_size], tf.float32, initializer=biases_initializer)
		out = tf.nn.bias_add(conv, var['b'])
		
		if batch_norm:
			out = tf.contrib.layers.batch_norm(out)
		if activation_fn != None:
			out = activation_fn(out)

		if synthetic:
			out_shape = out.get_shape().as_list()
			label_shape = label.get_shape().as_list()	# B, 10
			if conditional:
				label_tile = tf.reshape(tf.tile(label, [1,out_shape[1]*out_shape[2]]), [out_shape[0],out_shape[1],out_shape[2], label_shape[1]])
				out_syn = tf.concat(3, [out, label_tile])
			else:
				out_syn = out
			h1, var['l1_w'], var['l1_b'] = conv2d(out_syn, 128, [5,5], [1,1],
								tf.zeros_initializer, tf.zeros_initializer, batch_norm=True, activation_fn=tf.nn.relu, name='l1')
			h2, var['l2_w'], var['l2_b'] = conv2d(h1, 128, [5,5], [1,1],
								tf.zeros_initializer, tf.zeros_initializer, batch_norm=True, activation_fn=tf.nn.relu, name='l2')
			synthetic_grad, var['l3_w'], var['l3_b'] = conv2d(h2, 128, [5,5], [1,1],
								tf.zeros_initializer, tf.zeros_initializer, batch_norm=False, activation_fn=None, name='l3')
			return out, var['w'], var['b'], synthetic_grad
		else:		
			return out, var['w'], var['b']

def linear(inputs, output_size, 
			weights_initializer=initializers.xavier_initializer(),
			biases_initializer=tf.zeros_initializer,
			activation_fn=None, batch_norm=True, name='linear'):
	
	var = {}
	shape = inputs.get_shape().as_list()
	with tf.variable_scope(name):
		var['w'] = tf.get_variable('w', [shape[1], output_size], tf.float32,
						initializer=weights_initializer)
		var['b'] = tf.get_variable('b', [output_size],
						initializer=biases_initializer)
		out = tf.nn.bias_add(tf.matmul(inputs, var['w']), var['b'])

		if batch_norm:
			out = tf.contrib.layers.batch_norm(out)
		if activation_fn is not None:
			out = activation_fn(out)
		return out

def linear_w_variable(inputs, W, b, activation_fn=None, batch_norm=True, name='linear'):
	with tf.variable_scope(name):
		out = tf.nn.bias_add(tf.matmul(inputs, W), b)
		if batch_norm:
			out = tf.contrib.layers.batch_norm(out)
		if activation_fn is not None:
			out = activation_fn(out)
		return out

def get_time():
	return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


