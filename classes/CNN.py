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
import scipy as sp
import pickle


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
		
		#self.inv_P_ = np.load('/home/ar773/CIFARProcess/invP.npy')    #np.ones((3072,3072))
		#self.mean_ = np.load('/home/ar773/CIFARProcess/mean.npy')     #np.ones((3072,))
		self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
		
		self.indices = utils.indices
		self.batch_size = utils.batch_size
		self.stoch_bsize = utils.stoch_bsize		
		self.N = utils.N
		
		pass


	def load_orig(self):
		X = np.load('/home/ar773/CIFARProcess/X_before_gcn.npy').reshape((10000,3,32,32))
		self.Xt = np.array([X[i,:,:,:] for i in self.indices])
		print self.Xt.shape
		
		
	def set_data(self, Xt):
		self.Xt = Xt
		
		pass	
	
	def load(self, proto_path='', caffe_path=''):
		'''
		Load a model and it's corresponding bayesian one with a different prototxt file (with sample_weight:true)
		'''

		self.net = caffe.Net(proto_path+'.prototxt',caffe_path, caffe.TEST)
		self.net_bcnn = caffe.Net(proto_path+'_stoch.prototxt',caffe_path, caffe.TEST)
	
		pass

	def save_img_ind(self, X, path = 'images/img', tag = '', tr = True):
		Xt = X.copy()
		if tr == True:

			Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]*Xt.shape[3]))
		
			Xt = np.dot(Xt,self.inv_P_) + self.mean_
			Xt = Xt.reshape((Xt.shape[0],3,32,32))	
		for idx in xrange(self.N):
			sp.misc.imsave(path+'/images/img_'+str(self.indices[idx])+'_'+tag+'.jpg',np.rot90(Xt[idx].T,k=3))
	
	def load_db(self,dbtype='leveldb',dbno=1):
		'''
		Load the data base
		'''
		
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

			
			for key, _ in db:
				if count in self.indices:
					x, y = utils.get_cifar_image(db, str(key).zfill(5))
					Xt += [x]
					yt += [y]
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
			
			for _, value in lmdb_cursor:
				if count in self.indices:
					datum = caffe.proto.caffe_pb2.Datum()
					datum.ParseFromString(value)
					label = int(datum.label)
					image = caffe.io.datum_to_array(datum)
					image = image.astype(np.float32)
					Xt += [image]
					yt += [label]
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
			print 'NONZERO GRADS: ', np.count_nonzero(input_grads), '\t GSTEPS ',gstep
			input_fool -= input_grads* 1.e-2 
		
		'''
		save the images 
		'''
		np.save(src_path+'/results/nn_100cprob',c_prob)
		np.save(src_path+'/results/nn_100imgadv',img_adv)
		np.save(src_path+'/results/nn_100ytadv',yt_adv)
		np.save(src_path+'/results/nn_100yt',self.yt)
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
			#print 'Backprop Batch: ',b
			input_batch = img_set[partition,:,:,:]
			self.net.blobs[l_pen].diff[...] = prob[partition]
			self.net._backward(list(self.net._layer_names).index(l_pen), 0)
			img_grads[partition,:,:,:] = self.net.blobs[l_first].diff
			partition += self.batch_size*np.ones(self.batch_size,dtype='int64')
		return img_grads

	def get_data_grads_stoch(self,img_set,prob):
		'''
		get backpropagated gradients 
		'''
		l_first = 'data'
		l_last = list(self.net_bcnn._layer_names)[-1]
		l_pen = list(self.net_bcnn._layer_names)[-2]
		img_grads = img_set.copy()
		partition = np.arange(self.batch_size)

		if l_pen == 'ip2':
			pass
		else:
			prob = prob[:,:,None,None]

		for b in xrange(img_set.shape[0]/self.batch_size):
			print 'Backprop Batch: ',b
			input_batch = img_set[partition,:,:,:]
			self.net_bcnn.blobs[l_pen].diff[...] = prob[partition]
			self.net_bcnn._backward(list(self.net_bcnn._layer_names).index(l_pen), 0)
			img_grads[partition,:,:,:] = self.net_bcnn.blobs[l_first].diff
			partition += self.batch_size*np.ones(self.batch_size,dtype='int64')
		return img_grads

	def get_stoch_probs(self,img_set):
		'''
		get stochastic update for one imput image
		'''
		l_first = 'data'
		l_last = list(self.net_bcnn._layer_names)[-1]
		prob_bcnn = np.zeros((img_set.shape[0],self.stoch_bsize,10))
		for idx in xrange(img_set.shape[0]):
			input_bcnn = np.array([img_set[idx,:,:,:].copy() for _ in xrange(self.stoch_bsize)])
			self.net_bcnn.blobs[l_first].reshape(*input_bcnn.shape)
			self.net_bcnn.blobs[l_first].data[...] = input_bcnn
			self.net_bcnn.forward()
			prob_bcnn[idx,:,:] = self.net_bcnn.blobs[l_last].data.squeeze().copy()
		'''
		prob_bcnn.shape = (img_set.shape[0],stoch_bsize,10)
		'''
		return prob_bcnn

	def get_stoch_probs2(self,img_set):
		'''
		get stochastic probs
		'''
		l_first = 'data'
		l_last = list(self.net_bcnn._layer_names)[-1]
		prob_cnn = np.zeros((img_set.shape[0],10))
		partition = np.arange(self.batch_size)
		for b in xrange(img_set.shape[0]/self.batch_size):
			print 'Batch: ', b, '\t', partition[0]
			input_batch = img_set[partition,:,:,:]
			self.net_bcnn.blobs[l_first].reshape(*input_batch.shape)
			self.net_bcnn.blobs[l_first].data[...] = input_batch
			self.net_bcnn.forward()
			prob_cnn[partition,:] = self.net_bcnn.blobs[l_last].data.squeeze().copy()
			partition += self.batch_size*np.ones(self.batch_size,dtype='int64')

		'''
		prob_cnn.shape = img_set.shape[0],10)
		'''
		return prob_cnn

		
	def get_det_probs(self,img_set):
		'''
		get stochastic update for one imput image
		'''
		l_first = 'data'
		l_last = list(self.net._layer_names)[-1]
		prob_cnn = np.zeros((img_set.shape[0],10))
		partition = np.arange(self.batch_size)
		for b in xrange(img_set.shape[0]/self.batch_size):
			#print 'Batch: ', b, '\t', partition[0]
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
	
	def get_adv_stoch_probs(self, stoch_bsize = 100, grad_steps = 20, mode = 'trial'):

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
			prob = self.get_stoch_probs2(img_set=input_fool,stoch_bsize=1)

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
			print 'NONZERO GRADS: ', np.count_nonzero(input_grads), '\t GSTEPS ',gstep
			input_fool -= input_grads* 4.e-2 
		
		'''
		save the images 
		'''
		np.save(src_path+'/results/nncprob',c_prob)
		np.save(src_path+'/results/nnimgadv',img_adv)
		np.save(src_path+'/results/nnytadv',yt_adv)
		np.save(src_path+'/results/nnyt',self.yt)
		pass




