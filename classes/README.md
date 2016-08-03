This readme is for the documentation of calsses and class methods in the "classes" folder.


CNN.py

class CNN
	Members:
		model - an object of class Sequential, a CNN model for this experiment
		X_train - training set (eg. images in CIFAR-10)
		Y_train - training set labels (eg. the 10 classes of CIFAR-10)
		X_test - test set (eg. images in CIFAR-10)
 		Y_test - test set (eg. image labels in CIFAR-10)
	Methods:
		set_model_arch - defines the architecture for a model (number of layers, dimensions for each layer, and other specifications)
		train_model - compiles a model, defines an optimiser, fits a training set, saves a log file for train/test/validation error, saves the trained and model and the corresponding weights
		load_model - loads a saved model and it's corresponding weights
		save_img - saves an image (2-dim numpy array) as a jpeg at the specified path
		get_rnd_adv_img -  generates adversarial examples for a set of images, step-size and num_gradients specified as arguments
		get_rnd_adv_label - helper function to generate adversarial labels for a given set of corrected labels
		get_img - given the index number, get the corresponding image and label from either the training set or test set
		compute_test_error - evaluate a trained model by computing test-error for a given set of test images and their correct labels
		get_stats - 
		print_report - (STATIC METHOD)
		noise_adv - generate adversarial examples for a noisy image
		gen_adversarial -  
		
		
		
		
