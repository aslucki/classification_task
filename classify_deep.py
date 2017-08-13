from __future__ import print_function
from preprocessor import utils
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ConfigParser import SafeConfigParser
import argparse
import os, shutil
import h5py
import numpy as np

#Create command line arguments parser
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True, help='Path to the configuration file')
ap.add_argument('-e', '--extract', action='store_true', 
				help='Extract CNN codes from images')
ap.add_argument('-v', '--visualize', action='store_true', 
				help='Visualize CNN codes in 2 dimensions')
ap.add_argument('-t', '--train', action='store_true', help='Train and evaluate')
args = vars(ap.parse_args())


#Parse config file
parser = SafeConfigParser()
parser.read('config.ini')


#Execute if the script started with the -e option
if args['extract']:
	
	#Loading keras is time consuming, so it may be reasonable
	#to import it here, against the PEP
	from deep_clf.features_extractor import VGG16_layer
	
	#Local helper functions
	def prepare_datasets(file_path, features_ds, features_dim):
		data_file = h5py.File(file_path, mode='a')
		data_file.create_dataset(features_ds,(0, features_dim), 
								maxshape=(None, features_dim),dtype='float')
		return data_file
		
	def get_file(source_path, destination_path, ds_name, dims):
		if os.path.exists(destination_path):
			file = h5py.File(destination_path, 'a')
		else:
			shutil.copyfile(source_path, destination_path)
			file = prepare_datasets(destination_path, ds_name, dims)
		return file
		
	def extract_codes(data_array, batch_size=1000):
		batches = utils.get_batches(data_array,  batch_size)
		
		for batch in batches:
			print('[INFO] Processing next batch')
			features = extractor.compute(batch)
			utils.save_data(features, data_array)
		
	
	#Get selected layer name
	layer_name = parser.get('settings', 'vgg_layer')
	#Create extractor instance
	extractor = VGG16_layer(layer_name)
	#Get CNN codes size
	features_dim = extractor.model.get_layer(layer_name).output_shape[-1]
	
	#Get setting and paths
	vgg_training_path = parser.get('local_paths','training_vgg_features') 
	vgg_training_data = os.path.join(vgg_training_path, 'trn_vgg_{}.h5'.format(layer_name))
	vgg_testing_path = parser.get('local_paths','testing_vgg_features') 
	vgg_testing_data = os.path.join(vgg_testing_path, 'test_vgg_{}.h5'.format(layer_name))
	training_data = parser.get('local_paths', 'training_data')
	testing_data = parser.get('local_paths', 'testing_data')
	deep_ds = parser.get('datasets', 'deep')
	data_ds = parser.get('datasets', 'data')


	#--------------------#
	#Process training data
	#--------------------#
	
	#Get training data file
	trn_vgg_file = get_file(training_data, vgg_training_data, deep_ds, features_dim)
	#Compute codes from selected layer
	extract_codes(trn_vgg_file[data_ds])
	trn_vgg_file.close()
	
	#--------------------#
	#Process testing data
	#--------------------#
	
	#Get training data file
	test_vgg_file = get_file(testing_data, vgg_testing_data, deep_ds, features_dim)
	#Compute codes from selected layer
	extract_codes(test_vgg_file[data_ds])
	test_vgg_file.close()


if args['visualize']:
	
	#Get paths
	layer_name = parser.get('settings', 'vgg_layer')
	vgg_testing_path = parser.get('local_paths','testing_vgg_features') 
	vgg_testing_data = os.path.join(vgg_testing_path, 'test_vgg_{}.h5'.format(layer_name))
	
	#Get data
	data_file = h5py.File(vgg_testing_data, mode='r')
	deep_ds = parser.get('datasets', 'deep')
	labels_ds = parser.get('datasets', 'target')
	data = data_file[deep_ds][:3000]
	labels = data_file[labels_ds][:3000]
	
	reduced_tsne = TSNE(n_components=2, n_iter=1000).fit_transform(data)
	pca = PCA(n_components=2)
	reduced_pca = pca.fit_transform(data)
	
	plt.subplot(121)
	plt.title('Dimensions reduced with TSNE')
	plt.scatter(reduced_tsne[:,0], reduced_tsne[:,1], c=labels)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title('Dimensions reduced with PCA')
	plt.scatter(reduced_pca[:,0], reduced_pca[:,1], c=labels)
	plt.xticks([])
	plt.yticks([])
	
	plt.show()
	
	
	data_file.close()
	
if args['train']:
	
	#Get paths
	layer_name = parser.get('settings', 'vgg_layer')
	vgg_training_path = parser.get('local_paths','training_vgg_features') 
	vgg_training_data = os.path.join(vgg_training_path, 'trn_vgg_{}.h5'.format(layer_name))
	vgg_testing_path = parser.get('local_paths','testing_vgg_features') 
	vgg_testing_data = os.path.join(vgg_testing_path, 'test_vgg_{}.h5'.format(layer_name))
	
	#Get data
	deep_ds = parser.get('datasets', 'deep')
	labels_ds = parser.get('datasets', 'target')
	trn_vgg_file = h5py.File(vgg_training_data, mode='r')
	test_vgg_file = h5py.File(vgg_testing_data, mode='r')
	
	validation_size = 10000
	#(train_data, train_labels) = (trn_vgg_file[deep_ds][:-validation_size], 
	#							trn_vgg_file[labels_ds][:-validation_size])
	(validation_data, validation_labels) = (trn_vgg_file[deep_ds][:10], 
								trn_vgg_file[labels_ds][:10])						
	#(test_data, test_labels) = (test_vgg_file[deep_ds][:], test_vgg_file[labels_ds][:])
	
	
	print(validation_data.shape)
	trn_vgg_file.close()
	test_vgg_file.close()



