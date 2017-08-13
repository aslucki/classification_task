"""
Utilities for data preprocessing,
files operations and visualizations.
"""

from __future__ import print_function
import urllib
import tarfile
import os, shutil
from progressbar import Percentage, ProgressBar,Bar,ETA
import numpy as np
import cPickle
import h5py


def download_file(url, destination_path):
	"""Download a file from an url."""
	
	file_name = url.split('/')[-1]
	domain = url.split('/')[2]
	
	pbar = ProgressBar(widgets=[Bar(), ' ', Percentage(), ' ', ETA()])
	def dl_progress(count, block_size, total_size):
		pbar.update( int(count * block_size * 100 / (total_size+1e-7)) )
	
	print('[INFO] Downloading {} from {}'.format(file_name, domain))
	pbar.start()
	urllib.urlretrieve(url, destination_path, 
	                     reporthook=dl_progress)
	print('[INFO] File saved in {}'.format(destination_path))
	
	
def unpack_tar(source_path, destination_path):
	"""Extract contents of a tar file to a directory."""
	
	file_name = source_path.split('/')[-1]
	
	print('[INFO] Unpacking {} to {}'.format(file_name, destination_path))
	with tarfile.open(source_path) as tar:
		root_name = tar.getnames()[0]
		tar.extractall(destination_path)
	
	print('[INFO] Cleaning {} directory'.format(destination_path))
	#Move files destination path
	files_list = os.listdir(os.path.join(destination_path,root_name))
	for file in files_list:
		shutil.move(os.path.join(destination_path,root_name,file), 
					destination_path)
	
	#Delete empty root directory
	os.rmdir(os.path.join(destination_path, root_name))
	
	return files_list
	
	
	
def unpickle(pickle_file):
	"""Load data dictionary from a pickle file"""
    
	with open(pickle_file, 'rb') as file:
		data_dict = cPickle.load(file)
	return data_dict
	

def cvt_cifar(cifar_array):
	"""Reshape cifar data array to
	 (n, 32, 32, 3) and convert to BGR format.
	 
	 Args:
		cifar_array: Data array form CIFAR dictionary,
					cifar_dict['data']
	 
	 Returns:
		Array of images reshaped to (n, 32, 32, 3)
		in BGR format.
	"""

	n = np.asarray(cifar_array).shape[0]
	
	#Pixel values in CIFAR data are stacked vertically
	#in a single column
	R =  cifar_array[:,:1024].reshape(n,32,32)
	G =  cifar_array[:,1024:2048].reshape(n,32,32)
	B =  cifar_array[:,-1024:].reshape(n,32,32)
	
	#Stack channels in BGR order
	img_array = np.stack([B,G,R],axis=3)
	return img_array
	
def get_samples(data_array, labels, n_samples=10):
	"""Select n random elements of each category from data_array."""
	
	samples = []
	
	#Iterate through unique labels
	for i in set(labels):
		#Randomly select n indexes of data_array elements with a given label
		selected_data = np.random.choice(np.argwhere(labels == i).flatten(),
										n_samples)
		samples.append( data_array[selected_data] )
		
	return np.asarray(samples)
	
def create_mosaic(img_array, padding=2):
	"""Reshape array of images to 2D for displaying 
	as a single image.
	
	Args:
		img_array: array of shape (n_imgs, imgs_in_row, 
									img_height, img_width, n_channels)
									
	Returns:
		mosaic: single image consisting of images from img_array,
			optionally with a white padding
	"""

	#Add white padding to every image in the array
	with_padding = np.pad(img_array,((0,0),(0,0),(padding,padding),
	                      (padding,padding),(0,0)), 
	                      'constant', constant_values=(255))
	
	#Stack rows vertically
	mosaic = np.vstack(
						#Stack images horizontally to form rows
						[ np.hstack(with_padding[i][:]) 
						   for i in range( len(img_array) ) ]
					   )
	return mosaic
	
	
def save_data(data_array, h5_dataset):
	"""Resize HDF5 dataset and append data_array to it."""
	
	#Get length of the data_array
	data_size = np.asarray(data_array).shape[0]
	
	#Get length of the dataset
	dataset_end = h5_dataset.shape[0]
	
	#Resize the dataset to fit new data
	h5_dataset.resize( (dataset_end + data_size,) + h5_dataset.shape[1:] )
	
	#Append data to the dataset
	h5_dataset[dataset_end:dataset_end + data_size] = data_array

def get_batches(data_array, batch_size=1000):
	"""Generate batches from dataset"""
	
	start = 0
	while True:
		batch = data_array[start:start + batch_size]
		if len(batch) == 0: break
		
		yield batch
		
		if len(batch) < batch_size: break
		start += batch_size