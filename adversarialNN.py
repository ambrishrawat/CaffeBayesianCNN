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
	temp = out.reshape(num_images,10)
	temp = [softmax(x) for x in temp]
	return temp

def argmax(p):
	pass

#net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S_deploy.prototxt','/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S.caffemodel', caffe.TEST)

net = caffe.Net('models/train_val.prototxt','models/cifar10_nin.caffemodel', caffe.TEST)
print 'Data layer input shape:', net.blobs['data'].data.shape, 'Label layer input shape:', net.blobs['label'].data.shape,

db = plyvel.DB('./cifar-test-leveldb/')
#db = plyvel.DB('../data/cifar10_nin/cifar-test-leveldb')

Xt = []
yt = []
count = 0
for key, _ in db:
	#print key
	if count==4:
		x, y = get_cifar_image(db, str(key).zfill(5))
		Xt += [x]
		yt += [y]
		break
	count+=1
db.close()
Xt = np.array(Xt)
yt = np.array(yt)

print 'Xt shape: ', Xt.shape, ' Yt shape', yt, yt.shape
print list(net.blobs.keys())

'''
List of dictionary keys for the network

['data', 'label', 'label_cifar_1_split_0', 'label_cifar_1_split_1', 'conv1', 'cccp1', 'cccp2', 'pool1', 'conv2', 'cccp3', 'cccp4', 'pool2', 'conv3', 'cccp5', 'cccp6', 'pool3', 'pool3_pool3_0_split_0', 'pool3_pool3_0_split_1', 'accuracy', 'loss']

'''

for i in range(0,Xt.shape[0]):
	caffe_input = Xt
	net.blobs['data'].data[...] = caffe_input
	net.blobs['label'].data[...] = np.array([1])
	# make a prediction from the kitten pixels
	out = net.forward(end='accuracy')
	print 'out ',out

	#net_adv is the target label
	net_adv = 5

	#different threshold probabilities to check how much perturbation is required to get a particular adversarial classifictation probability 
	target_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

	T = 10 #(number of classes)
	caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]
	for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):
		net.blobs['data'].data[...] = caffe_input_fooled
		out = net.forward(end='pool3')['pool3']
		out = get_proabability_vector(out)
		print 'prob ',np.sum(out)
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
