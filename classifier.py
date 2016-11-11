import tensorflow as tf
import numpy as np
import math
import os
from utils import linear_w_variable, unpickle, conv2d, pooling
from tensorflow.contrib.layers.python.layers import initializers
import pdb

class Model():

	def __init__(self, sess, dataset, conf, num_train=50000, input_size=3072, test_filename='/data2/andrewliao11/cifar-10-batches-py/test_batch'):

		self.sess = sess
		self.test_filename = test_filename
		self.w = 32
		self.h = 32
		self.channel = 3
		self.synthetic = conf.synthetic
		self.optim_type = conf.optim_type
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
		self.cDNI = conf.conditional
		self.dataset = dataset
		self.init_lr = conf.init_lr
		self.var = {}
		'''
		if self.optim_type == 'exp_decay':
			decay_steps = int(math.floor(self.num_train/self.batch_size)* conf.num_epoch_per_decay)
			self.lr = tf.train.exponential_decay(conf.init_lr,
							self.global_step, decay_steps,
							conf.decay_factor,
							staircase=True)
			self.optim = tf.train.GradientDescentOptimizer(self.lr)
			self.optim_grad = tf.train.GradientDescentOptimizer(self.lr)
		elif self.optim_type == 'adam':
		'''
	def init_variable(self, info, name):
		if 'linear' in name:
			with tf.variable_scope(name):
				W = tf.get_variable('w', [info['input_size'], info['output_size']], initializer=info['weights_initializer'])
				b = tf.get_variable('b', info['output_size'], initializer=info['biases_initializer'])
			return W, b

	def synthetic_linear_model(self, inputs, labels, conditional, 
					weights_initializer=initializers.xavier_initializer(),
					biases_initializer=tf.zeros_initializer, name='synthetic'):
		var = {}
		with tf.variable_scope(name):
			if conditional:
				inputs_syn = tf.concat(1, [inputs, labels])
			else:
				inputs_syn = tf.identity(inputs)

			shape = inputs_syn.get_shape().as_list()
			info = {'input_size':shape[1], 'output_size':4000, 'weights_initializer':tf.zeros_initializer,
							'biases_initializer':tf.zeros_initializer}
			W1, b1 = self.init_variable(info, name='l1_linear')
			h1 = linear_w_variable(inputs_syn, W1, b1, activation_fn=tf.nn.relu, batch_norm=True, name='l1')
			info = {'input_size':4000, 'output_size': shape[1], 'weights_initializer':tf.zeros_initializer,
							'biases_initializer':tf.zeros_initializer}
			W2, b2 = self.init_variable(info, name='l2_linear')
			synthetic_grad = linear_w_variable(h1, W2, b2, activation_fn=tf.nn.relu, batch_norm=True, name='l2') 
			return synthetic_grad

	def build_mlp_model(self):
		
		# hidden_layers
		self.layers_keys = ['l1', 'l2', 'l3', 'l4']
		self.imgs = tf.placeholder('float32',[self.batch_size, self.input_dims])
		self.labels = tf.placeholder('int32', [self.batch_size])
		self.labels_onehot = tf.one_hot(self.labels, self.output_size, on_value=1.0, off_value=0.0)
		# declare variable
		info = {'input_size':self.input_dims, 'output_size':self.hidden_size, 
						'weights_initializer':None,
						'biases_initializer':tf.zeros_initializer}
		self.var['l1_W'], self.var['l1_b'] = self.init_variable(info, name='l1_linear')
		info['input_size'] = self.hidden_size
		self.var['l2_W'], self.var['l2_b'] = self.init_variable(info, name='l2_linear')
		self.var['l3_W'], self.var['l3_b'] = self.init_variable(info, name='l3_linear')
		info['output_size'] = self.output_size
		self.var['l4_W'], self.var['l4_b'] = self.init_variable(info, name='l4_linear')
		# build graph
		h1 = linear_w_variable(self.imgs, self.var['l1_W'], self.var['l1_b'], 
								activation_fn=tf.nn.relu, batch_norm=True, name='l1')
		h2 = linear_w_variable(h1, self.var['l2_W'], self.var['l2_b'], 
								activation_fn=tf.nn.relu, batch_norm=True, name='l2')
		h3 = linear_w_variable(h2, self.var['l3_W'], self.var['l3_b'], 
								activation_fn=tf.nn.relu, batch_norm=True, name='l3')
		h4 = linear_w_variable(h3, self.var['l4_W'], self.var['l4_b'], 
								activation_fn=tf.nn.relu, batch_norm=True, name='l4')

		self.out_logit = tf.nn.softmax(h4)
 		self.out_argmax = tf.argmax(self.out_logit, 1)
		self.loss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(h4, self.labels)
		self.loss = tf.reduce_mean(self.loss_entropy)
		if self.synthetic:
			self.backprop_gradient = {}
			for k in self.layers_keys:
				self.backprop_gradient[k] = tf.gradients(self.loss, eval(k.replace('l','h')))

	def build_decoupled_mlp_model(self):
		self.layer_in = {}
		self.synthetic_gradient = {}
		self.layer_out = {}
		self.synthetic_loss = {}
		self.backprop_placeholder = {}
		self.synthetic_placeholder = {}
	
		for idx, k in enumerate(self.layers_keys):
			# layer k
			if k == 'l1':
				self.layer_in[k] = tf.placeholder('float32', [self.batch_size, self.input_dims])
			else:
				self.layer_in[k] = tf.placeholder('float32', [self.batch_size, self.hidden_size])
			W = self.var[k+'_W']
			b = self.var[k+'_b']
			self.layer_out[k] = linear_w_variable(self.layer_in[k], W, b, activation_fn=tf.nn.relu, 
									batch_norm=True, name='DNI_'+k+'_linear')
			if self.synthetic:
				if k == 'l4':
					self.synthetic_placeholder[k] = tf.placeholder('float32', [self.batch_size, self.output_size])
				else:
					self.synthetic_placeholder[k] = tf.placeholder('float32', [self.batch_size, self.hidden_size])
				self.synthetic_gradient[k] = self.synthetic_linear_model(self.layer_out[k],self.labels_onehot,
											self.cDNI, name=k+'_synthetic')

		if self.synthetic:
			self.synthetic_total_loss = 0
			for k in self.layers_keys:
				self.backprop_placeholder[k] = tf.placeholder('float32', self.synthetic_gradient[k].get_shape())
				self.synthetic_loss[k] = tf.reduce_mean(tf.square(self.synthetic_gradient[k]-self.backprop_placeholder[k]))
				self.synthetic_total_loss += self.synthetic_loss[k]

	def train(self):

		self.global_step_gradient = {}
		self.train_op_gradient = {}
		self.optim_gradient = {}
		summary_writer = tf.train.SummaryWriter('./tmp', self.sess.graph)
		# assign a optimizer to a stnthetic layer
		if self.synthetic:
			for k in self.layers_keys:
				self.global_step_gradient[k] = tf.get_variable('global_step_'+k+'_gradient', [],
													initializer=tf.constant_initializer(0), trainable=False)
				self.optim_gradient[k] = tf.train.AdamOptimizer(self.init_lr)
				vars = []
				for var in tf.trainable_variables():
					if k+'_synthetic' in var.name:
						vars.append(var)
				self.train_op_gradient[k] = self.optim_gradient[k].minimize(self.synthetic_loss[k], 
															global_step=self.global_step_gradient[k], var_list=vars)
		self.global_step = {}
		self.optim = {}
		self.train_op = {}
		# find the normal trainable variable
		for k in self.layers_keys:
			vars = []
			for var in tf.trainable_variables():
				if ('synthetic' not in var.name) and (k in var.name):
					vars.append(var)
			self.global_step[k] = tf.get_variable('global_step_'+k, [],initializer=tf.constant_initializer(0), trainable=False)
			self.optim[k] = tf.train.AdamOptimizer(self.init_lr)
			grads_and_vars = self.optim[k].compute_gradients(self.layer_out[k], vars)
			synthetic_grads_and_vars = []
			for grad, var in grads_and_vars:
				synthetic_grad = tf.gradients(self.layer_out[k],var,self.synthetic_placeholder[k])
				synthetic_grads_and_vars.append((synthetic_grad[0], var))
			self.train_op[k] = self.optim[k].apply_gradients(synthetic_grads_and_vars, global_step=self.global_step[k])

		tf.initialize_all_variables().run()
		self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)
		backprop_gradient = []
		for k in self.layers_keys:
			backprop_gradient.append(self.backprop_gradient[k])

		for epoch_idx in range(int(self.max_epoch)):
			for idx in range(int(math.floor(self.num_train/self.batch_size))):
				img_batch, label_batch = self.dataset.sequential_sample(self.batch_size)
				if self.synthetic:
					feed_dict = {self.imgs:img_batch, self.labels:label_batch}
					target_gradient = self.sess.run(backprop_gradient+[self.loss], feed_dict)
					loss = target_gradient.pop(-1)
					grad_loss = 0.
					for idx,l in enumerate(self.layers_keys):
						if idx == 0:
							feed_dict = {self.layer_in[l]: img_batch, self.labels: label_batch}
						else:
							feed_dict = {self.layer_in[l]: layer_out_prev, self.labels: label_batch}
						# forward the synthetic model
						feed_dict[self.backprop_placeholder[l]] = np.squeeze(target_gradient[idx])
						#### something wrong with here
						_, synthetic_loss, synthetic_gradient = self.sess.run([self.train_op_gradient[l], 
													self.synthetic_loss[l], self.synthetic_gradient[l]], feed_dict)
						grad_loss += synthetic_loss
						feed_dict[self.synthetic_placeholder[l]] = synthetic_gradient
						_, layer_out_prev = self.sess.run([self.train_op[l], self.layer_out[l]], feed_dict)
					print "[*] Iter {}, syn_grad_loss={}, real_loss={}".format(int(self.global_step['l1'].eval()), grad_loss, loss)
				else:
					_, loss = self.sess.run([self.train_op, self.loss],{
								self.imgs: img_batch,
								self.labels: label_batch
								})
					print "[*] Iter {}, real_loss={}".format(int(self.global_step.eval()), loss)

				if self.global_step['l1'].eval()%self.test_per_iter == 0 or self.global_step['l1'].eval()==1:
					self.evaluate(split='train')
					self.evaluate(split='test')

	def evaluate(self, imgs=None, labels=None, split='test'):
	
		if split == 'test':
			imgs, labels = self.dataset.random_sample(-1, phase='test')
		elif split == 'train':
			imgs, labels = self.dataset.random_sample(10000, phase='train')

		num_test = imgs.shape[0]
		correct = 0.
		test_imgs = 0.
		avg_loss = 0.
		for idx in range(int(math.floor(num_test/self.batch_size))):
			img_batch = imgs[idx*self.batch_size:(idx+1)*self.batch_size]
			label_batch = labels[idx*self.batch_size:(idx+1)*self.batch_size]
			pred, loss = self.sess.run([self.out_argmax, self.loss],{
						self.imgs: img_batch,
						self.labels: label_batch
						})
			correct_batch = self.calc_top1(pred, label_batch)
			correct += correct_batch
			test_imgs += img_batch.shape[0]
			avg_loss += loss

		print '[+] Top1 {} accuracy = {}, loss = {}'.format(split, correct/test_imgs, avg_loss/math.floor(num_test/self.batch_size))
		self.save_model()

	def calc_top1(self, pred, label):
		correct = np.sum((pred==label)+0.)
		return correct

	def save_model(self, name='checkpoint'):
		if not os.path.exists(self.ckpt_dir):
			os.makedirs(self.ckpt_dir)
		self.saver.save(self.sess, os.path.join(self.ckpt_dir, name), global_step=int(self.global_step['l1'].eval()))



