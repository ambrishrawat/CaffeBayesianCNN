import caffe 

caffe.set_device(0)
caffe.set_mode_gpu()

src_path = '/home/ar773/CaffeBayesianCNN/exps/'
models_path = '/home/ar773/CaffeBayesianCNN/models'

from classes.CNN import CNN
import classes.utils as utils
import numpy as np

def exp_adv(model = 'nodrop', dbtype = 'leveldb', dbno = 1, mode = 'trial'):
	'''
	get adversarial images
	'''	
	#model = 'alldrop'
	#model = 'fcdrop'	
	model = 'nodrop'
	stoch_bsize = 100
	grad_steps = 1000
	
	caffe_path = models_path + '/'+model+'/lenet_'+model+'_iter_100000.caffemodel'
	proto_path = models_path + '/'+model+'/lenet_'+model+'_deploy'
	cnn = CNN()
	cnn.load(proto_path=proto_path, caffe_path=caffe_path)
	cnn.load_db(mode=mode,dbtype=dbtype,dbno=dbno)
	c_prob1 = np.zeros((cnn.N,grad_steps, 10))

	model = 'alldrop'
	
	caffe_path = models_path + '/'+model+'/lenet_'+model+'_iter_100000.caffemodel'
	proto_path = models_path + '/'+model+'/lenet_'+model+'_deploy'
	cnn2 = CNN()
	cnn2.load(proto_path=proto_path, caffe_path=caffe_path)
	c_prob2 = np.zeros((cnn.N,grad_steps, 10))
	mc_prob2 = np.zeros((cnn.N,grad_steps, stoch_bsize,10))
	
	model = 'fcdrop'
	
	caffe_path = models_path + '/'+model+'/lenet_'+model+'_iter_100000.caffemodel'
	proto_path = models_path + '/'+model+'/lenet_'+model+'_deploy'
	cnn3 = CNN()
	cnn3.load(proto_path=proto_path, caffe_path=caffe_path)
	c_prob3 = np.zeros((cnn.N,grad_steps, 10))
	mc_prob3 = np.zeros((cnn.N,grad_steps, stoch_bsize,10))

	cnn2.set_data(cnn.Xt)
	cnn3.set_data(cnn.Xt)

	'''
	get adversarial label array correspondint to yt
	'''
	yt_adv = utils.get_adv_label(cnn.yt)
	'''
	c_prob : self.N x grad_step x stoch_bsize x probs_10
	img_adv : self.N x grad_steps x 3 x 32 x 32
	'''
	tr_ = [cnn.label_names[y] for y in cnn.yt]
	adv_ = [cnn.label_names[y] for y in yt_adv]
	
	img_adv = np.zeros((cnn.N, grad_steps, 3, 32, 32))

	input_fool = cnn.Xt.copy()
		
	for gstep in xrange(grad_steps):
		'''
		Step 1: Set the data for the network for which you want the adversarial image
		'''
		prob = cnn3.get_det_probs(img_set=input_fool)
		c_prob3[:,gstep,:] = prob.copy()

		
		'''
		Step 2: Set the probablisties for adversarial label
		'''
		corr = 0.0
		inn =0.0
		for idx in xrange(cnn.N):
			corr += np.mean(prob[:,cnn.yt[idx]])
			inn += np.mean(prob[:,yt_adv[idx]])
			prob[:,yt_adv[idx]] -= 1.
			
		print gstep, '\t', corr, '\t', inn
		

		prob1 = cnn.get_det_probs(img_set=input_fool)
		c_prob1[:,gstep,:] = prob1.copy()

		prob_stoch = cnn2.get_stoch_probs(img_set=input_fool,stoch_bsize=100)
		mc_prob2[:,gstep,:,:] = prob_stoch.copy()

		
		prob_ap = cnn2.get_det_probs(img_set=input_fool)
		c_prob2[:,gstep,:] = prob_ap.copy()
		

		prob_stoch = cnn3.get_stoch_probs(img_set=input_fool,stoch_bsize=100)
		mc_prob3[:,gstep,:,:] = prob_stoch.copy()
		'''
		prob_ap = cnn3.get_det_probs(img_set=input_fool)
		c_prob3[:,gstep,:] = prob_ap.copy()
		'''		
		'''
		Step 3: Backprop and add gradients
		'''
		input_grads = cnn3.get_data_grads(input_fool,prob)
		img_adv[:,gstep,:,:,:] = input_fool.copy()
		#print 'NONZERO GRADS: ', np.count_nonzero(input_grads), '\t GSTEPS ',gstep
		input_fool -= input_grads*9.e-4
		cnn3.save_img_ind(input_fool, path = src_path+'/exp3', tag = str(gstep))
		
	'''
	save the images 
	'''
	np.save(src_path+'/exp3'+'/results/nodrop_prob',c_prob1)
	np.save(src_path+'/exp3'+'/results/alldrop_ap_prob',c_prob2)
	np.save(src_path+'/exp3'+'/results/fcdrop_ap_prob',c_prob3)
	np.save(src_path+'/exp3'+'/results/alldrop_prob',mc_prob2)
	np.save(src_path+'/exp3'+'/results/fcdrop_prob',mc_prob3)
	np.save(src_path+'/exp3'+'/results/img_adv',img_adv)

	print tr_, adv_


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
	cnn.get_adv_probs(stoch_bsize = 100, grad_steps = 100)



if __name__ == "__main__":
	
	#exp1(model = 'nn', dbtype = 'leveldb', dbno = 1, mode ='full')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='full')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	
	exp2(model = 'nn', dbtype = 'leveldb', dbno = 1, mode ='trial')
