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
		self.batch_size = 100
		if mode=='trial':
			indices = [1,10,100,150,42,21,75,57,37,111, 234 ,542, 356 ,653,567]
			#indices = [1,10,100,150,42]
			#indices = [13,42]
			self.batch_size = 5
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


	def get_adv_probs(self, stoch_bsize = 100, grad_steps = 20, mode = 'trial'):

		'''
		get adversarial label array correspondint to yt
		'''
		yt_adv = utils.get_adv_label(self.yt)

		'''
		c_prob : self.N x grad_step x stoch_bsize x probs_10
		img_adv : self.N x grad_steps x 3 x 32 x 32
		'''
		
		c_prob = np.zeros((self.N,grad_steps, stoch_bsize,10))
		img_adv = np.zeros((self.N, grad_steps, 3, 32, 32))


		input_fool = self.Xt.copy()
			
		for gstep in xrange(grad_steps):

			'''
			Step 1: Set the data for the network for which you want the adversarial image
			'''
			prob = self.get_det_probs(img_set=input_fool)

			'''
			Step 2: Copy the data to the stochastic-network to get uncertainity estimates 
			'''
			prob_stoch = self.get_stoch_probs(img_set=input_fool,stoch_bsize=stoch_bsize)
			c_prob[:,gstep,:,:] = prob_stoch.copy()
			
			'''
			Step 3: Set the probablisties for adversarial label
			'''
			for idx in xrange(self.N):
				prob[:,yt_adv[idx]] -= 1.
					
			'''
			Step 4: Backprop and add gradients
			'''
			input_grads = self.get_data_grads(input_fool,prob)
			img_adv[:,gstep,:,:,:] = input_fool.copy()
			input_fool -= input_grads* 4.e1
		
		'''
		save the images 
		'''
		np.save('cprob',c_prob)
		np.save('imgadv',img_adv)
		pass


	def get_data_grads(self,img_set,prob):
		'''
		get backpropagated gradients 
		'''
		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		l_pen = list(self.net._layer_names)[-2]
		img_grads = img_set.copy()
		partition = np.arange(self.batch_size)

		if l_pen == 'ip2':
			pass
		else:
			prob = prob[:,:,None,None]

		for b in xrange(img_set.shape[0]/self.batch_size):
			print 'Backprop Batch: ',b
			input_batch = img_set[partition,:,:,:]
			self.net.blobs[l_pen].diff[...] = prob[partition]
			self.net._backward(list(self.net._layer_names).index(l_pen), 0)
			img_grads[partition,:,:,:] = self.net.blobs[l_first].diff
			partition += self.batch_size*np.ones(self.batch_size,dtype='int64')
		return img_grads

	def get_stoch_probs(self,img_set, stoch_bsize=100):
		'''
		get stochastic update for one imput image
		'''
		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		prob_bcnn = np.zeros((img_set.shape[0],stoch_bsize,10))
		for idx in xrange(img_set.shape[0]):
			input_bcnn = np.array([img_set[idx,:,:,:].copy() for _ in xrange(stoch_bsize)])
			self.net_bcnn.blobs[l_first].reshape(*input_bcnn.shape)
			self.net_bcnn.blobs[l_first].data[...] = input_bcnn
			self.net_bcnn.forward()
			prob_bcnn[idx,:,:] = self.net_bcnn.blobs[l_last].data.squeeze().copy()
		'''
		prob_bcnn.shape = (img_set.shape[0],stoch_bsize,10)
		'''
		return prob_bcnn
		
	def get_det_probs(self,img_set):
		'''
		get stochastic update for one imput image
		'''
		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		prob_cnn = np.zeros((img_set.shape[0],10))
		partition = np.arange(self.batch_size)
		for b in xrange(img_set.shape[0]/self.batch_size):
			print 'Batch: ', b, '\t', partition[0]
			input_batch = img_set[partition,:,:,:]
			self.net.blobs[l_first].reshape(*input_batch.shape)
			self.net.blobs[l_first].data[...] = input_batch
			self.net.forward()
			prob_cnn[partition,:] = self.net.blobs[l_last].data.squeeze().copy()
			partition += self.batch_size*np.ones(self.batch_size,dtype='int64')

		'''
		prob_cnn.shape = img_set.shape[0],10)
		'''
		return prob_cnn

	def get_accuracy(self):
		'''
		Accuracy (argmax)
		'''
		
		prob = self.get_det_probs(self.Xt)
		y_out = [prob[i,:].argmax() for i in xrange(self.N)]
		y_out = np.array(y_out).reshape(*self.yt.shape)
		
		acc = float(np.count_nonzero(y_out==self.yt))/float(self.N)
		print acc
	
		pass
		

	def get_accuracy_bcnn(self,stoch_bsize = 100):
		'''
		Accuracy (argmax)
		'''
		
		probs = self.get_stoch_probs(self.Xt, stoch_bsize=stoch_bsize)
		prob_mean = np.mean(probs,axis=1)
		
		y_out = [prob_mean[i,:].argmax() for i in xrange(self.N)]
		y_out = np.array(y_out).reshape(*self.yt.shape)
		
		acc = float(np.count_nonzero(y_out==self.yt))/float(self.N)
		print acc
	
		pass
	




