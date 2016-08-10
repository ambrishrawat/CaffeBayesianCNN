import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
import plyvel
import pickle
import pylab

#Global parameters shared across all models, lenel-none, lenet-ip-std, lenet-all-std, lenet-ip-mc and lenet-all-mc

indices =  np.load('/home/ar773/a.npy')[0:100]
indices = np.array([1])
#indices = np.arange(10000)
N = indices.shape[0]
		#for full-mode set N to 10000 
#indices = np.load('/home/ar773/CaffeBayesianCNN/classes/indices.npy')[0:N]		#random permutation of 10000
#batch_size = 2	#batch size for lenet-none, lenel-ip-std, lenel-all-std
batch_size = 1
stoch_bsize = 100				#mc for lenet-ip-mc and lenet-all-mc

#s = np.random.normal(-8.2984457782616281e-09,0.27796287433968719,(100,3,32,32))

#-2.0016396251849e-08

#0.27795847357736214

def gen_noise(mean=0.0,var=0.1):
	sigma = var**0.5
	gauss = np.random.normal(mean,sigma,(3,32,32))
	gauss = gauss.reshape(ch,row,col)
	return gauss


#indices = [2,875] 2.3e-2
#

gcn_normalizer = np.load('/home/ar773/CaffeBayesianCNN/normalizers.npy')
gcn_normalizer = np.array([ gcn_normalizer[i] for i in indices])

gcn_mean = np.load('/home/ar773/CaffeBayesianCNN/mean_gcn.npy')
gcn_mean = np.array([ gcn_mean[i] for i in indices])

def backward_T(input_grads,inv_P_,mean_):
	Xt = input_grads.copy()
	Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]*Xt.shape[3]))		
	Xt = np.dot(Xt,inv_P_) + mean_
	Xt *= gcn_normalizer[:,np.newaxis]
	Xt = Xt + gcn_mean[:, np.newaxis]
	Xt = Xt.reshape((Xt.shape[0],3,32,32))
	return Xt

def forward_T(input_orig,P_,mean_):
	Xt = input_orig.copy()
	Xt = Xt.reshape((Xt.shape[0],Xt.shape[1]*Xt.shape[2]*Xt.shape[3]))
	Xt = Xt - gcn_mean[:, np.newaxis]
	Xt /= gcn_normalizer[:,np.newaxis]
	Xt = Xt - mean_
	Xt = np.dot(Xt,P_)
	Xt = Xt.reshape((Xt.shape[0],3,32,32))
	return Xt
def softmax(w, t = 1.0):
	'''
	compute softmax
	'''
	e = np.exp(w/t)
	dist = e / np.sum(e)
	return dist

def get_cifar_image(db, key):
	'''
	required for getting an (image,label) tuple from a lmdb data base
	'''
	raw_datum = db.get(key)
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)
	flat_x = np.array(datum.float_data)
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label
	return x, y


def get_adv_label(yt):
	'''
	get adversarial label
	'''
	yt_adv = yt.copy()
	yt_adv = np.array([(y+1)%10 for y in yt_adv])
	
	return yt_adv 	
