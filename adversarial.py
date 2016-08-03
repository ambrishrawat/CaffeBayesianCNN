import caffe 

caffe.set_device(0)
caffe.set_mode_gpu()

models_path = '/home/ar773/CaffeBayesianCNN/models'

from classes.CNN import CNN

def exp1():
	'''
	load a net 
	load cifar
	predict class for a random image 
	'''
	model = 'nn'
	if model=='zoo':
		caffe_path = models_path + '/modelZooNN/cifar10_nin.caffemodel'
		proto_path = models_path + '/modelZooNN/train_val'
	else:
		caffe_path = models_path + '/bcnn/lenet_all_dropout_iter_100000.caffemodel'
		proto_path = models_path + '/bcnn/lenet_all_dropout_sampleTest_deploy'
		
	cnn = CNN()
	cnn.load(proto_path=proto_path, caffe_path=caffe_path)

	if model=='zoo':
		cnn.load_db(dbtype='leveldb')
	else:
		cnn.load_db(dbtype='leveldb')
	pass

	cnn.get_adv_class(model=model)


if __name__ == "__main__":
	print 'Import successfull'
	exp1()
	
