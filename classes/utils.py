import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import lmdb
from collections import defaultdict
import plyvel
import pickle
import pylab

def softmax(w, t = 1.0):
	'''
	compute softmax
	'''
	e = np.exp(w/t)
	dist = e / np.sum(e)
	return dist

def get_cifar_image(db, key):
	'''
	required for getting an (image,label) tuple from a lmdb data base
	'''
	raw_datum = db.get(key)
	datum = caffe.proto.caffe_pb2.Datum()
	datum.ParseFromString(raw_datum)
	flat_x = np.array(datum.float_data)
	x = flat_x.reshape(datum.channels, datum.height, datum.width)
	y = datum.label
	return x, y


def get_adv_label(yt):
	'''
	get adversarial label
	'''
	yt_adv = yt.copy()
	yt_adv = np.array([(y+1)%10 for y in yt_adv])
	
	return yt_adv 	
