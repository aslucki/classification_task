"""
Download data from cifar webpage,
unpack the downloaded tar file,
convert data to BGR format and save to HDF5.
Visualize sample images from the data.
"""

from __future__ import print_function
from preprocessor import utils
from ConfigParser import SafeConfigParser
import argparse
import os, glob
import h5py
import cv2

#Create command line arguments parser
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', required=True, help='Path to the configuration file')
ap.add_argument('-d', '--download', action='store_true', help='Download data from external source')
ap.add_argument('-u', '--unpack', action='store_true', help='Extract downloaded data')
ap.add_argument('-p', '--preprocess', action='store_true', help='Preprocess data and save to HDF5')
ap.add_argument('-s', '--samples', nargs='?', type=int, choices=[10,20,30,50], const=True, 
				action='store', help='Show and save n sample images from each category')
args = vars(ap.parse_args())

#Parse config file
parser = SafeConfigParser()
parser.read(args['config'])

#Set local name for external file
cifar_file = 'cifar_data.tar.gz'

#Execute if the script started with the -d option
if args['download']:
	url = parser.get('external_paths', 'cifar10_url')
	destination = parser.get('local_paths', 'raw_data')
	
	utils.download_file(url, os.path.join(destination, cifar_file))


	
	
#Execute if the script started with the -u option
if args['unpack']:
	raw_data_path = parser.get('local_paths', 'raw_data')
	source_path = os.path.join(raw_data_path, cifar_file)
	
	#Check if the file exists
	if os.path.exists(source_path):
		utils.unpack_tar(source_path, raw_data_path)
	else:
		print('[WARNING] File {} does not exist in {}'.format(cifar_file, raw_data_path))
		


	

#Execute if the script started with the -p option
if args['preprocess']:
	
	#Define local helper functions
	def prepare_datasets(file_path,features_ds, labels_ds):
		data_file = h5py.File(file_path, mode='w')
		data_file.create_dataset(features_ds, (0,32,32,3), maxshape=(None,32,32,3),
					dtype='uint8')
		data_file.create_dataset(labels_ds, (0,), maxshape=(None,),
					dtype='uint8')
	
		return data_file
		
	def process_cifar(file):
		cifar_dict = utils.unpickle(file)
		img_array = utils.cvt_cifar(cifar_dict['data'])
		utils.save_data(img_array, data_file[features_ds])
		utils.save_data(cifar_dict['labels'], data_file[labels_ds])
	
	#Get datasets names
	features_ds = parser.get('datasets', 'data')
	labels_ds = parser.get('datasets', 'target')
	
	#--------------------#
	#Process training data
	#--------------------#
	
	training_data = parser.get('local_paths', 'training_data')	
	
	#Check if hdf5 file with training data exists
	if os.path.exists(training_data):
		data_file = h5py.File(training_data, mode='a')
	else:
		data_file = prepare_datasets(training_data, features_ds, labels_ds)
		
	#Get paths to pickle files with training data
	raw_data_path = parser.get('local_paths', 'raw_data')
	training_files = glob.glob(os.path.join(raw_data_path, 'data_batch_*') )
	
	#Convert and save training data to hdf5 file
	print('[INFO] Saving training data to {}'.format(training_data))
	for file in training_files:
		process_cifar(file)
		print('   [INFO] Saved {} rows to {}'.format(data_file[features_ds].shape[0], training_data))
	data_file.close()
	

	#--------------------#
	#Process testing data
	#--------------------#
	
	testing_data = parser.get('local_paths', 'testing_data')
	
	#Check if hdf5 file with testing data exists
	if os.path.exists(testing_data):
		data_file = h5py.File(testing_data, mode='a')
	else:
		data_file = prepare_datasets(testing_data, features_ds, labels_ds)
	
	#Get path to a picke file with testing data
	testing_file = os.path.join(raw_data_path, 'test_batch') 
	
	#Convert and save testing data to hdf5 file
	print('[INFO] Saving testing data to {}'.format(testing_data))
	process_cifar(testing_file)
	print('[INFO] Saved {} rows to {}'.format(data_file[features_ds].shape[0], testing_data))
	data_file.close()
	
	
	
#Execute if the script started with the -s option
if args['samples']:
	
	#Max accepted value is 50
	n = args['samples']
	
	#Get paths to necessary files
	data_path = parser.get('local_paths', 'training_data')
	figures_path = parser.get('local_paths', 'figures')
	
	#Get datasets names
	features_ds = parser.get('datasets', 'data')
	labels_ds = parser.get('datasets', 'target')
	
	#Check if HDF5 file with training data exists
	if os.path.exists(data_path):
		file = h5py.File(data_path,'r')
		
		'''
		As stated on the CIFAR webpage, every pickle file contains roughly 
		the same number of images from each category.
		Therefore it's safe to select random images only from the subset of the data.
		'''
		#Get 10000 images and labels from data
		data = file[features_ds][:10000]
		labels = file[labels_ds][:10000]
		file.close()
		
		#Get random images from data
		samples = utils.get_samples(data, labels, n)
		#Create 2D image from sample images
		mosaic = utils.create_mosaic(samples)
		
		#Save image to disk
		cv2.imwrite(os.path.join(figures_path, 'samples.png'), mosaic)
		print('[INFO] Saved image to {}'.format(figures_path))
		
		#Show generated image
		cv2.imshow("Sample images", mosaic)
		cv2.waitKey(0)
	else:
		print('[WARNING] {} does not exist'.format(data_path))
	
		
	
	
	