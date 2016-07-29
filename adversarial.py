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

def softmax(w, t = 1.0):
	e = np.exp(w/t)
	dist = e / np.sum(e)
	return dist

def get_cifar_image(db, key):
	raw_datum = db.get(key)
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)
	flat_x = np.array(datum.float_data)
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label
	return x, y

def get_proabability_vector(out):
	num_images = out.shape[0]
	temp = out.reshape(10)
	return temp

#load the network (lenet_all_dropout_train_test.prototxt)
net = caffe.Net('/home/ar773/CaffeBayesianCNN/bcnn/lenet_all_dropout_deploy.prototxt','/home/ar773/CaffeBayesianCNN/bcnn/lenet_all_dropout_iter_100000.caffemodel', caffe.TEST)



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
	image = image.astype(np.float32)

	accumulate = []

	print label
	adv_label = 6

	print 'Data layer input shape:', net.blobs['data'].data.shape
	caffe_input = np.asarray([image])
	print 'Image shape', caffe_input.shape
	

	net.blobs['data'].reshape(*caffe_input.shape)
	net.blobs['data'].data[...] = caffe_input
	net.forward()

	target_probs = [0.1] #[0.1,0.2,0.3,0.4,0.5,0.6]
	caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]

	for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):


		adv_prob = 0.0
		temp_prob = []		

		for _ in xrange(100):
		#while adv_prob < target_prob:

			#net.blobs['data'].reshape(*caffe_input_fooled.shape)
			net.blobs['data'].data[...] = caffe_input_fooled
			net.forward()
			prob = net.blobs['softmax'].data.copy()
			#print prob
			temp = prob.reshape((10,)).argmax()
			#print temp,label
			adv_prob = get_proabability_vector(prob)
			adv_prob = adv_prob[adv_label]
			print adv_prob
			
			accumulate.append(adv_prob)
			prob[:,adv_label] = -1.
			net.blobs['ip2'].diff[...] = prob[:,None,None,:]
			np.count_nonzero(prob)	
			net._backward(list(net._layer_names).index('ip2'), 0)
			#print np.count_nonzero(net.blobs['softmax'].diff)	
			caffe_input_fooled -= net.blobs['data'].diff * 1e-1	
	break	
pylab.plot(accumulate)
pylab.show()
