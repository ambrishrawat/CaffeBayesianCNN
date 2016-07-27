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

def get_cifar_image(db, key):
	raw_datum = db.get(key)
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)
	flat_x = np.array(datum.float_data)
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label
	return x, y

#net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S_deploy.prototxt','/home/ar773/CaffeBayesianCNN/modelVGG/VGG_CNN_S.caffemodel', caffe.TEST)

net = caffe.Net('models/train_val.prototxt','models/cifar10_nin.caffemodel', caffe.TEST)
print 'Data layer input shape:', net.blobs['data'].data.shape

db = plyvel.DB('./cifar-test-leveldb/')
#db = plyvel.DB('../data/cifar10_nin/cifar-test-leveldb')

Xt = []
yt = []
for key, _ in db:
	#print key
	x, y = get_cifar_image(db, str(key).zfill(5))
	Xt += [x]
	yt += [y]
	break
db.close()
Xt = np.array(Xt)
yt = np.array(yt)

print 'image shape: ', Xt.shape
net.blobs['data'].data[...] = Xt
output = net.forward(end='pool3')
print output['pool3']



