import tensorflow as tf
from utils import linear

# assume the input vector is 1D vector
class synthetic_grad():

	def __init__(self, inputs, name='synthetic_grad'):

		self.var = {}
		# B, H
		self.inputs_shape = inputs.get_shape()
		with tf.variable_scope(name):
			self.h1, self.var['l1_w'], self.var['l1_b'] = linear(inputs, 4000, weights_initializer=tf.zeros_initializer, 
												biases_initializer=tf.zeros_initializer, activation_fn=tf.nn.relu, batch_norm=True, name='l1')
			self.synthetic_grad, self.var['l2_w'], self.var['l2_b'] = linear(self.h1, self.inputs_shape[1], weights_initializer=tf.zeros_initializer,
												biases_initializer=tf.zeros_initializer, activation_fn=tf.nn.relu, batch_norm=True, name='l2')



