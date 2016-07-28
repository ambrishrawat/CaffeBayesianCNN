import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
#caffe.set_device(0)
caffe.set_mode_cpu()

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


#net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S_deploy.prototxt','/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S.caffemodel', caffe.TEST)

net = caffe.Net('models/train_val.prototxt','/home/ar773/CaffeBayesianCNN/models/cifar10_nin.caffemodel', caffe.TEST)
print 'Data layer input shape:', net.blobs['data'].data.shape

db = plyvel.DB('./cifar-test-leveldb/')


Xt = []
yt = []
count = 0
for key, _ in db:
	#print key
	if count==12 or count==120:
		x, y = get_cifar_image(db, str(key).zfill(5))
		Xt += [x]
		yt += [y]
		
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
adv_label = 1


for i in xrange(0,Xt.shape[0]-1):
	caffe_input = Xt[i].reshape(1,3,32,32)
	net.blobs['data'].reshape(*caffe_input.shape)
	net.blobs['data'].data[...] = caffe_input
	net.forward()

	target_probs = [0.6] #[0.1,0.2,0.3,0.4,0.5,0.6]
	caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]
	for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):


		adv_prob = 0.0
		temp_prob = []		

		for _ in xrange(1):
		#while adv_prob < target_prob:

			net.blobs['data'].reshape(*caffe_input_fooled.shape)
			net.blobs['data'].data[...] = caffe_input_fooled
			net.forward(data=caffe_input_fooled)
			prob = net.blobs['softmax'].data.copy()

			adv_prob = get_proabability_vector(prob)
			#print adv_prob
			adv_prob = adv_prob[adv_label]
			prob[:,adv_label] = 1.
			#net.blobs['pool3'].diff[...] = prob[:,None,None,:]
			print prob
			net.backward(softmax=prob)
			#print net.blobs['pool3'].diff
			caffe_input_fooled -= net.blobs['data'].diff * 1e2
	
		
		'''
		highest_ind = prob.argmax()
		new_ind_prob = prob[:,new_ind].min()
		i = 0
		while new_ind_prob < target_prob:
		prob = net2.forward(data=caffe_input_fooled, label=np.zeros((T,1,1,1)))['prob']
		prob[:,new_ind] -= 1.
		net2.blobs['fc8'].diff[...] = prob[:,None,None,:]
		net2._backward(list(net2._layer_names).index('fc8'), 0)
		caffe_input_fooled -= net2.blobs['drop0'].diff * 1e2
		prob = net.forward(data=caffe_input_fooled)['prob']
		highest_ind = prob.argmax()
		new_ind_prob = prob[:,new_ind].min()
		i += 1
		if i % 1 == 0:
			print net2.blobs['drop0'].diff.sum()
			print highest_ind
			print new_ind_prob
			prob = net2.forward(data=caffe_input_fooled, label=np.zeros((T,1,1,1)))['prob']
			print prob.argmax()
			print prob[:,new_ind]
			print
		'''




