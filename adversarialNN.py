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


#load the network (lenet_all_dropout_train_test.prototxt)
net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelAllDropout/lenet_all_dropout_train_test.prototxt','/home/ar773/CaffeBayesianCNN/modelAllDropout/cifar10_uncertainty_data/lenet_all_dropout_iter_100000.caffemodel', caffe.TEST)

#load the network (lenet_all_dropout_sampleTest_deploy.prototxt)
net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelAllDropout/lenet_all_dropout_sampleTest_deploy.prototxt','/home/ar773/CaffeBayesianCNN/modelAllDropout/cifar10_uncertainty_data/lenet_all_dropout_iter_100000.caffemodel', caffe.TEST)

#load the lmbd data set
lmdb_env = lmdb.open('/home/ar773/packages/caffe/examples/cifar10/cifar10_test_lmdb/')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#check the dictionary of layers
print list(net.blobs.keys())

'''
List of dictionary keys for the network

['data', 'label', 'label_cifar_1_split_0', 'label_cifar_1_split_1', 'conv1', 'dropC1', 'pool1', 'conv2', 'dropC2', 'pool2', 'ip1', 'relu1', 'drop1', 'ip2', 'ip2_ip2_0_split_0', 'ip2_ip2_0_split_1', 'accuracy', 'loss']
'''

for key, value in lmdb_cursor:

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)

	label = int(datum.label)
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)

	print label

	print 'Data layer input shape:', net.blobs['data'].data.shape
	caffe_input = np.asarray([image])
	print 'Image shape', caffe_input.shape
	

	net.blobs['data'].data[...] = caffe_input
	# make a prediction from the kitten pixels
	out = net.forward(end='softmax')
	print out

	# extract the most likely prediction
	print("Predicted class is #{}.".format(out['ip2'][0].argmax()))
	#input image
	
	#net_adv is the target label
	net_adv = 5

	#different threshold probabilities to check how much perturbation is required to get a particular adversarial classifictation probability 
	target_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

	T = 10 #(number of classes)
	caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]
	for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):
		prob = net.forward(data=caffe_input_fooled, end='softmax')['softmax']
		highest_ind = prob.argmax()
		new_ind_prob = prob[:,new_ind].min()
		i = 0

		while new_ind_prob < target_prob:
			prob = net.forward(data=caffe_input_fooled, label=np.zeros((T,1,1,1)))['prob']
		
		prob[:,new_ind] -= 1.

		net.blobs['data'].diff[...] = prob[:,None,None,:]
		net._backward(list(net._layer_names).index('data'), 0)

		#add gradients with respect to the 'data' layer
		caffe_input_fooled -= net.blobs['data'].diff * 1e2

		prob = net.forward(data=caffe_input_fooled, end='softmax')['softmax']
		highest_ind = prob.argmax()
		new_ind_prob = prob[:,new_ind].min()
		i += 1
		if i % 1 == 0:
			print net.blobs[''].diff.sum()
			print highest_ind
			print new_ind_prob
			prob = net2.forward(data=caffe_input_fooled, label=np.zeros((T,1,1,1)))['prob']
			print prob.argmax()
			print prob[:,new_ind]
			print
