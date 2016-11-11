import tensorflow as tf
import os
from data_loader import cifar
import pprint
import classifier
import numpy as np
flags = tf.app.flags
pp = pprint.PrettyPrinter().pprint

tf.app.flags.DEFINE_integer('max_step', 50000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('model_name', 'cnn', 'model used here')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'save the ckpt model')
tf.app.flags.DEFINE_string('gpu_fraction', '1/3', 'define the gpu fraction used')
tf.app.flags.DEFINE_integer('batch_size', 256, '')
tf.app.flags.DEFINE_integer('test_batch_size', 256, '')
tf.app.flags.DEFINE_integer('hidden_size', 1000, '')
tf.app.flags.DEFINE_integer('output_size', 10, '')
tf.app.flags.DEFINE_integer('test_per_iter', 500, '')
tf.app.flags.DEFINE_integer('max_to_keep', 20, '')
tf.app.flags.DEFINE_string('optim_type', 'adam', '[exp_decay, adam]')
tf.app.flags.DEFINE_boolean('allow_growth', True, '')
tf.app.flags.DEFINE_boolean('synthetic', False, 'use synthetic gradients or not')
tf.app.flags.DEFINE_boolean('conditional', False, 'use conditional DNI or not')
# exponetial decay
'''
tf.app.flags.DEFINE_float('init_lr', 0.1, '')
tf.app.flags.DEFINE_float('decay_factor', 0.1, '')
tf.app.flags.DEFINE_integer('num_epoch_per_decay', 350, '')
'''
# adam optimizer
tf.app.flags.DEFINE_float('init_lr', 3e-5, '')

conf = flags.FLAGS
# set random seed
append = lambda x: '/media/VSlab2/andrewliao11_data/cifar-10-batches-py/data_batch_'+x
train_filename = [ append(str(i+1)) for i in range(5)]
test_filename = '/media/VSlab2/andrewliao11_data/cifar-10-batches-py/test_batch'

def calc_gpu_fraction(fraction_string):
	idx, num = fraction_string.split('/')
	idx, num = float(idx), float(num)

	fraction = 1 / (num - idx + 1)
	print "[*] GPU : %.4f" % fraction
	return fraction

def main(_):

	attrs = conf.__dict__['__flags']
	pp(attrs)
	conf.checkpoint_dir = os.path.join(conf.checkpoint_dir, conf.model_name)
	dataset = cifar()

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = calc_gpu_fraction(conf.gpu_fraction)	
	if conf.allow_growth:
		config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		model = classifier.Model(sess, dataset, conf, num_train=dataset.num_train, input_size=dataset.input_size)
		if conf.model_name == 'cnn':
			model.build_cnn_model()
		elif conf.model_name == 'mlp':
			model.build_mlp_model()
		else:
			assert()
		if conf.synthetic:
			model.build_decoupled_mlp_model()
		model.train()	

if __name__ == '__main__':
	tf.app.run()

