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


#load the network
net = caffe.Net('/home/ar773/CaffeBayesianCNN/modelAllDropout/lenet_all_dropout_train_test.prototxt','/home/ar773/CaffeBayesianCNN/modelAllDropout/cifar10_uncertainty_data/lenet_all_dropout_iter_100000.caffemodel', caffe.TEST)



#load the lmbd data set
lmdb_env = lmdb.open('/home/ar773/packages/caffe/examples/cifar10/cifar10_test_lmdb/')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()


#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

print list(net.blobs.keys())
print 'yoda'
for key, value in lmdb_cursor:
	print 'yoda2'

	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(value)


	label = int(datum.label)
	image = caffe.io.datum_to_array(datum)
	image = image.astype(np.uint8)

	print label
	
	out = net.forward_all(data=np.asarray([image]))
	print out
	break
	plabel = int(out['prob'][0].argmax(axis=0))
	print plabel
	break
	count += 1
	iscorrect = label == plabel
	correct += (1 if iscorrect else 0)
	matrix[(label, plabel)] += 1
	labels_set.update([label, plabel])
	if not iscorrect:
		print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
		sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
		sys.stdout.flush()


'''
target_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
caffe_input_fooled_probs = [caffe_input.copy() for _ in xrange(len(target_probs))]
for target_prob, caffe_input_fooled in zip(target_probs, caffe_input_fooled_probs):
    prob = net.forward(data=caffe_input_fooled)['prob']
    highest_ind = prob.argmax()
    new_ind_prob = prob[:,new_ind].min()
    i = 0
    #while highest_ind != new_ind:
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
