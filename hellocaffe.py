import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe

caffe.set_device(0)

caffe.set_mode_gpu()

net = caffe.Net('conv.prototxt', caffe.TEST)
print net.blobs['conv'].data.shape

print net.inputs

im = np.array(Image.open('cat_gray.jpg'))
im_input = im[np.newaxis, np.newaxis, :, :]
net.blobs['data'].reshape(*im_input.shape)
net.blobs['data'].data[...] = im_input

out = net.forward()

print out
