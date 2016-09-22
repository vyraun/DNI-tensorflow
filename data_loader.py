import numpy as np
import utils

class cifar():

	'''
	Using CIFAR10
	data download in :https://www.cs.toronto.edu/~kriz/cifar.html
	data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a
        32x32 colour image. The first 1024 entries contain the red channel
        values, the next 1024 the green, and the final 1024 the blue. The
        image is stored in row-major order, so that the first 32 entries of
         the array are the red channel values of the first row of the image.
	labels -- a list of 10000 numbers in the range 0-9. The number at index i
        indicates the label of the ith image in the array data.
	'''
	def __init__(self):

		append = lambda x: '/data2/andrewliao11/cifar-10-batches-py/data_batch_'+x
		self.train_filename = [ append(str(i+1)) for i in range(5)]
		self.test_filename = '/data2/andrewliao11/cifar-10-batches-py/test_batch'
		self.num_train = 50000
		self.num_test = 10000
		self.input_size = 3072
		self.imgs = np.zeros([self.num_train, self.input_size], dtype='float32')
		self.labels = np.zeros([self.num_train], dtype='int32')
		self.current = 0
		for i in range(5):
			data_batch = utils.unpickle(self.train_filename[i])
			self.imgs[i*10000:(i+1)*10000] = data_batch['data']/255.
			self.labels[i*10000:(i+1)*10000] = np.asarray(data_batch['labels'])
		data_batch = utils.unpickle(self.test_filename)	
		self.test_imgs = data_batch['data']/255.
		self.test_labels = np.asarray(data_batch['labels'])

	def random_sample(self, batch_size, phase='train'):
		
		if phase == 'train':
			index = np.arange(self.num_train)
			np.random.shuffle(index)
			imgs = self.imgs[index]
			labels = self.labels[index]
		elif phase == 'test':
			index = np.arange(self.num_test)
			np.random.shuffle(index)
			imgs = self.test_imgs[index]
			labels = self.test_labels[index]

		if batch_size == -1:
			return imgs, labels	
		else:
			return imgs[:batch_size], labels[:batch_size]

	def sequential_sample(self, batch_size):
		end = (self.current+batch_size)%self.num_train
		if self.current + batch_size < self.num_train:
			imgs = self.imgs[self.current:(self.current+batch_size)%self.num_train]
			labels = self.labels[self.current:(self.current+batch_size)%self.num_train]	
		else:
			imgs = np.concatenate([self.imgs[self.current:], self.imgs[:end]], axis=0)
			labels = np.concatenate([self.labels[self.current:], self.labels[:end]], axis=0)
		self.current = end
		return imgs, labels
			

