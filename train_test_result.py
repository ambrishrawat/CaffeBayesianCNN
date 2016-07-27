#Program to compute and plot the error of the whole test and training set. 

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

caffemodel = 1000

while caffemodel <=100000:

	#load the network
	net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelAllDropout/lenet_all_dropout_train_test.prototxt','/home/ar773/CaffeBayesianCNN/modelAllDropout/cifar10_uncertainty_data/lenet_all_dropout_iter_'+str(caffemodel)+'.caffemodel', caffe.TEST)

	out = net.forward()
	print out
	
	#TEST SET
	#load the lmbd data set
	lmdb_env = lmdb.open('/home/ar773/packages/caffe/examples/cifar10/cifar10_test_lmdb/')
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe.proto.caffe_pb2.Datum()

	
