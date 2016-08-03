import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
caffe.set_device(0)
caffe.set_mode_gpu()

import plyvel
import pickle
import pylab
import utils
import random

src_path = '/home/ar773/CaffeBayesianCNN'

class CNN:
	'''
	CNN Class
	'''
	def __init__(self):
		#self.net
		#self.net_bcnn
		#self.Xt
		#self.yt
		#self.N		
		pass


	def load(self, proto_path='', caffe_path=''):
		'''
		Load a model and it's corresponding bayesian one with a different prototxt file (with sample_weight:true)
		'''

		self.net = caffe.Net(proto_path+'.prototxt',caffe_path, caffe.TEST)
		self.net_bcnn = caffe.Net(proto_path+'_bcnn.prototxt',caffe_path, caffe.TEST)
	
		pass


	def load_db(self,mode='trial',dbtype='leveldb',dbno=1):
		'''
		Load the data base
		'''
		indices = []
		if mode=='trial':
			indices = [1,10,100,150,42,21,75,57,37,111]
		if (dbtype=='leveldb'):
			if dbno==1:
				db = plyvel.DB(src_path+'/data/cifar10_gcn-leveldb/cifar-test-leveldb/')
			elif dbno==2:
				db = plyvel.DB(src_path+'/data/cifar10_gcn_padded-leveldb/cifar-test-leveldb/')
			else:
				db = plyvel.DB(src_path+'/data/cifar-test-leveldb/')
			count = 0
			Xt = []
			yt = []
			self.N = 0

			
			for key, _ in db:
				if count in indices or mode=='full':
					x, y = utils.get_cifar_image(db, str(key).zfill(5))
					Xt += [x]
					yt += [y]
					self.N += 1
				count+=1			
			db.close()
			self.Xt = np.array(Xt)
			self.yt = np.array(yt)

		elif (dbtype=='lmdb'):
			lmdb_env = lmdb.open('/home/ar773/packages/caffe/examples/cifar10/cifar10_test_lmdb/')
			lmdb_txn = lmdb_env.begin()
			lmdb_cursor = lmdb_txn.cursor()
			datum = caffe.proto.caffe_pb2.Datum()
			count = 0
			Xt = []
			yt = []
			
			self.N = 0
			for _, value in lmdb_cursor:
				if count in indices or mode=='full':
					datum = caffe.proto.caffe_pb2.Datum()
					datum.ParseFromString(value)
					label = int(datum.label)
					image = caffe.io.datum_to_array(datum)
					image = image.astype(np.float32)
					Xt += [image]
					yt += [label]
					self.N +=1
				count+=1
			self.Xt = np.array(Xt)
			self.yt = np.array(yt)

		else:
			print 'UNKNOWN TYPE'
	
		print 'Data loaded successfully'
		print 'Input(shape): ', self.Xt.shape, ' Labels(shape): ', self.yt.shape, ' N:', self.N

	def get_adv_class(self, num_samples = 100, num_iter = 20):


		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		l_pen = list(self.net._layer_names)[-2]

 
		print 'First: ', l_first, '\tLast: ', l_last, '\tPenultimate: ', l_pen
		adv_label = 1
		num_samples = 100
		num_iter = 100


		x = []
		c_prob = []


		caffe_input = self.Xt.reshape(self.N,3,32,32)
		target_probs = [0.6] #[0.1,0.2,0.3,0.4,0.5,0.6]
		caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]

		for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):

			for c_iter in xrange(num_iter):

				'''
				Step 1: Set the data for the network for which you want the adversarial image
				'''			
				self.net.blobs[l_first].data[...] = caffe_input_fooled
				self.net.forward()
				prob = self.net.blobs[l_last].data.squeeze().copy()
				if len(prob.shape) == 1:
					prob = prob[None]

				'''
				Step 2: Copy the data to the stochastic-network to get uncertainity estimates 
				'''
				caffe_input_fooled_bcnn = np.array([caffe_input_fooled.copy() for _ in xrange(num_samples)]).reshape(self.N*num_samples,3,32,32)
				self.net_bcnn.blobs[l_first].data[...] = caffe_input_fooled_bcnn
				self.net_bcnn.forward()
				prob_bcnn = self.net_bcnn.blobs[l_last].data.squeeze().copy()
				print prob_bcnn.shape
			

				'''
				Step 3: Copy the predicted probs in appropirate arrays for plotting
				'''
				adv_prob = prob_bcnn[:,adv_label]
				c_prob = np.append(c_prob,adv_prob)
				x = np.append(x,c_iter*np.ones((num_samples,)))



				'''
				Step 4: Backprop the error
					Update the loss/the derivative for the penultimate layer
					Take care of the tensor shapes for the layers (depends on the architecture)
				'''
				prob[:,adv_label] -= 1.
				#print prob.shape
				#print 'YODA', self.net.blobs[l_pen].diff.shape
				self.net.blobs[l_pen].diff[...] = prob[:,None,None,:]
				self.net._backward(list(self.net._layer_names).index(l_pen), 0)
				caffe_input_fooled -= self.net.blobs[l_first].diff * 4.e1

		'''
		Plot the obtained probabilities for classification 
		'''
		c_prob = np.array(c_prob)
		x = np.array(x)
		print 'YODA', c_prob.shape ,x.shape
		pylab.scatter(x,c_prob)
		pylab.show()


	def get_accuracy(self,dbtype='leveldb',dbno=1, mode = 'trial'):

		'''
		Load test-set
		'''
		self.load_db(mode=mode,dbtype=dbtype,dbno=dbno)
		
		'''
		Check accuracy (deterministic NN)

			Step1: Reshape the input data blob 
			Step2: One forward pass
		'''

		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		l_pen = list(self.net._layer_names)[-2]

		batch_size = 100
		num_correct = 0.0
		for b in xrange(self.N/batch_size):
		
			input_batch = self.Xt[b:b+batch_size,:,:,:]
			print 'Input shape (from file)', self.net.blobs[l_first].data.shape
			self.net.blobs[l_first].reshape(*input_batch.shape)
			self.net.blobs[l_first].data[...] = input_batch
			print 'Input shape (after reshape)', self.net.blobs[l_first].data.shape
			self.net.forward()

			prob = self.net.blobs[l_last].data.copy()
			print prob.shape #(batch_size,10,1,1)
			
			y_out = [prob[i,:,:,:,].argmax() for i in xrange(batch_size)]
			y_out = np.array(y_out).reshape(*self.yt[b:b+batch_size].shape)
	
			num_correct += np.count_nonzero(y_out==self.yt[b:b+batch_size])
			
		acc = float(num_correct)/float(self.N)
		print acc
	
		pass
		
	




