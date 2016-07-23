import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
caffe.set_device(0)
caffe.set_mode_gpu()
	
net = caffe.Net('models/train_val.prototxt','models/cifar10_nin.caffemodel', caffe.TEST)

count = 0
correct = 0
matrix = defaultdict(int) # (real,pred) -> int
labels_set = set()

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

