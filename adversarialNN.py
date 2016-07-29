import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
#caffe.set_device(0)
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

def get_proabability_vector(out,numsamples):
	temp = out.reshape(10,numsamples)
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
num_samples = 100
num_iter = 1

c_prob = np.zeros((num_iter,num_samples))

for i in xrange(0,Xt.shape[0]-1):
	caffe_input = Xt[i].reshape(1,3,32,32)


	target_probs = [0.6] #[0.1,0.2,0.3,0.4,0.5,0.6]
	caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(num_samples)]
	for target_prob in target_probs:
		caffe_input_fooled = np.array(caffe_input_fooled_probs).reshape(num_samples,3,32,32)
		adv_prob = 0.0
		temp_prob = []		

		for c_iter in xrange(num_iter):
		#while adv_prob < target_prob:

			print net.blobs['data'].data.shape, 'input data shape ', caffe_input_fooled.shape
			#net.blobs['data'].reshape(*caffe_input_fooled.shape)
			net.blobs['data'].data[...] = caffe_input_fooled
			net.forward()
			prob = net.blobs['softmax'].data.copy()
			print prob.shape
			
			adv_prob = get_proabability_vector(prob,num_samples)
			
						
			adv_prob = adv_prob[adv_label,:].reshape((num_samples,))
			print 'yoda', adv_prob
			c_prob[c_iter,:] = adv_prob
			prob[:,adv_label] -= 1.
			#print net.blobs['pool3'].diff.shape
			print adv_prob
			net.blobs['pool3'].diff[...] = prob[:,None,None,:]
			net._backward(list(net._layer_names).index('pool3'), 0)			
			#print net.blobs['pool3'].diff
			#net.backward(softmax=prob)
			#print np.count_nonzero(net.blobs['softmax'].diff)	
			#print net.blobs['pool3'].diff
			#print net.blobs['pool3'].diff
			caffe_input_fooled -= net.blobs['data'].diff * 1e-2
	
		
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

pylab.plot(accumulate)
pylab.show()


