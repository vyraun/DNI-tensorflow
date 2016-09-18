import tensorflow as tf
import numpy as np
import math
import os
from utils import linear
from tensorflow.contrib.layers.python.layers import initializers
import pdb

class mlp():

	def __init__(self, sess, conf, num_train=50000, input_size=3072):

		self.sess = sess
		self.test_per_iter = conf.test_per_iter
		self.max_step = conf.max_step
		self.ckpt_dir = conf.checkpoint_dir
		self.batch_size = conf.batch_size
		self.num_train = num_train
		self.max_epoch = math.floor(conf.max_step/math.floor(self.num_train/self.batch_size))
		self.input_dims = input_size
		self.hidden_size = conf.hidden_size
		self.weight_initializer = initializers.xavier_initializer()
		self.bias_initializer = tf.constant_initializer(0.1)
		self.output_size = conf.output_size
		self.max_to_keep = conf.max_to_keep
		self.var = {}

		self.global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
		decay_steps = int(math.floor(self.num_train/self.batch_size)* conf.num_epoch_per_decay)
		self.lr = tf.train.exponential_decay(conf.init_lr,
							self.global_step, decay_steps,
							conf.decay_factor,
							staircase=True)
		self.optim = tf.train.GradientDescentOptimizer(self.lr)

	def build_model(self):
		
		self.imgs = tf.placeholder('float32',[self.batch_size, self.input_dims])

		self.h1, self.var['l1_w'], self.var['l1_b'] = linear(self.imgs, self.hidden_size, 
							self.weight_initializer, self.bias_initializer, activation_fn=tf.nn.relu, name='l1_linear')
		self.h2, self.var['l2_w'], self.var['l2_b'] = linear(self.h1, self.hidden_size,
							self.weight_initializer, self.bias_initializer, activation_fn=tf.nn.relu, name='l2_linear')
		self.h3, self.var['l3_w'], self.var['l3_b'] = linear(self.h2, self.hidden_size, 
							self.weight_initializer, self.bias_initializer, activation_fn=tf.nn.relu, name='l3_linear')
		self.out, self.var['l4_w'], self.var['l4_b'] = linear(self.h3, self.output_size, 
							self.weight_initializer, self.bias_initializer, activation_fn=tf.nn.relu, name='l4_linear')
		self.out_logit = tf.nn.softmax(self.out)
		self.out_argmax = tf.argmax(self.out_logit, 1)
		self.labels = tf.placeholder('int32', [self.batch_size])
		self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.out, self.labels)
		self.loss = tf.reduce_sum(self.loss)/self.batch_size

	def train(self, imgs, labels):

		tf.initialize_all_variables().run()
		self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
		self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)
		for epoch_idx in range(int(self.max_epoch)):
			index = np.arange(self.num_train)
			np.random.shuffle(index)
			imgs = imgs[index]
			labels = labels[index]
			for idx in range(int(math.floor(self.num_train/self.batch_size))):
				img_batch = imgs[idx*self.batch_size:(idx+1)*self.batch_size]
				label_batch = labels[idx*self.batch_size:(idx+1)*self.batch_size]
				_, loss = self.sess.run([self.train_op, self.loss],{
							self.imgs: img_batch,
							self.labels: label_batch
							})
				print "Iter {}, loss={}".format(int(self.global_step.eval()), loss)
				if self.global_step.eval()%self.test_per_iter == 0:
					self.save_model()
	
	#def test(self)
	def save_model(self, name='checkpoint'):
		if not os.path.exists(self.ckpt_dir):
			os.makedirs(self.ckpt_dir)
		self.saver.save(self.sess, os.path.join(self.ckpt_dir, name), global_step=int(self.global_step.eval()))


