import caffe 

caffe.set_device(0)
caffe.set_mode_gpu()

models_path = '/home/ar773/CaffeBayesianCNN/models'

from classes.CNN import CNN


def exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode = 'trial'):
	'''
	Compute accuracy on the test set
	'''
	
	if model=='zoo':
		caffe_path = models_path + '/modelZooNN/cifar10_nin.caffemodel'
		proto_path = models_path + '/modelZooNN/train_val'
	else:
		caffe_path = models_path + '/bcnn/lenet_all_dropout_iter_100000.caffemodel'
		proto_path = models_path + '/bcnn/lenet_all_dropout_sampleTest_deploy'

	cnn = CNN()
	cnn.load(proto_path=proto_path, caffe_path=caffe_path)
	cnn.load_db(mode=mode,dbtype=dbtype,dbno=dbno)
	#cnn.get_accuracy()
	cnn.get_accuracy_bcnn(stoch_bsize = 1000)

def exp2(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode = 'trial'):
	'''
	Get one adversarial image and save the plot	
	'''
	
	if model=='zoo':
		caffe_path = models_path + '/modelZooNN/cifar10_nin.caffemodel'
		proto_path = models_path + '/modelZooNN/train_val'
	else:
		caffe_path = models_path + '/bcnn/lenet_all_dropout_iter_100000.caffemodel'
		proto_path = models_path + '/bcnn/lenet_all_dropout_sampleTest_deploy'
		
	cnn = CNN()
	cnn.load(proto_path=proto_path, caffe_path=caffe_path)
	cnn.load_db(mode=mode,dbtype=dbtype,dbno=dbno)
	cnn.get_adv_probs(stoch_bsize = 100, grad_steps = 50)



if __name__ == "__main__":
	
	#exp1(model = 'nn', dbtype = 'leveldb', dbno = 1, mode ='full')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='full')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	
	exp2(model = 'zoo', dbtype = 'leveldb', dbno = 3, mode ='trial')
