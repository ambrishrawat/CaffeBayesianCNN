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

db = plyvel.DB('./cifar-test-leveldb/')
#db = plyvel.DB('../data/cifar10_nin/cifar-test-leveldb')

Xt = []
yt = []
for key, _ in db:
	#print key
	x, y = get_cifar_image(db, str(key).zfill(5))
	Xt += [x]
	yt += [y]
db.close()
Xt = np.array(Xt)
yt = np.array(yt)

output = net.forward(Xt)
print output
#with open('./cifar10_gcn.pkl', 'wb') as handle:
#	pickle.dump([Xt, yt], handle)



'''
count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

db = plyvel.DB('./cifar-test-leveldb/')
for key,value in db:
	print key

lmdb_env = lmdb.open('./cifar-test-leveldb/')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

print 'yoda'
print lmdb_cursor
for key, value in lmdb_cursor:
	print 'yoda2'
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)


	label = int(datum.label)
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)
	out = net.forward_all(data=np.asarray([image]))
	plabel = int(out['prob'][0].argmax(axis=0))
	print plabel
	count += 1
	iscorrect = label == plabel
	correct += (1 if iscorrect else 0)
	matrix[(label, plabel)] += 1
	labels_set.update([label, plabel])
	if not iscorrect:
		print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
		sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
		sys.stdout.flush()

print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
print ""
print "Confusion matrix:"
print "(r , p) | count"
for l in labels_set:
	for pl in labels_set:
		print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])
'''
