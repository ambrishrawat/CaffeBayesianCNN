import caffe 

caffe.set_device(0)
caffe.set_mode_gpu()

from exps.exp3.exp import exp_adv

if __name__ == "__main__":
	
	#exp1(model = 'nn', dbtype = 'leveldb', dbno = 1, mode ='trial')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='full')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	#exp1(model = 'zoo', dbtype = 'leveldb', dbno = 1, mode ='trial')
	
	#exp2(model = 'nn', dbtype = 'leveldb', dbno = 1, mode ='trial')
	exp_adv(model = 'nodrop', dbtype = 'leveldb', dbno = 3, mode = 'trial')
