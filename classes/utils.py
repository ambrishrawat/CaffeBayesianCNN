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

N = 200				#for full-mode set N to 10000 
indices = np.load('/home/ar773/CaffeBayesianCNN/classes/indices.npy')[0:N]		#random permutation of 10000
batch_size = 100			#batch size for lenet-none, lenel-ip-std, lenel-all-std
stoch_bsize = 100				#mc for lenet-ip-mc and lenet-all-mc

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
