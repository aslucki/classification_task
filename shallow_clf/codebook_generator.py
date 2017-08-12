from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
from progressbar import Percentage, ProgressBar,Bar,ETA
import numpy as np
import cPickle


def get_random_subset(features, sample_percent):
	"""Select subset of the data to form a codebook"""
	
	features_size = np.asarray(features).shape[0]
	
	#Calculate number of samples based on the percentage
	n_samples = int(np.ceil(sample_percent * features_size))
	
	#Select random indexes
	idxs = np.random.choice(np.arange(0, features_size), (n_samples), replace=False)
	
	#Sort for easier access
	idxs.sort()
	
	data = []
	
	print('[INFO] Retrieving {} random samples from dataset'.format(n_samples))
	pbar = ProgressBar(widgets=[Bar(), ' ', Percentage(), ' ', ETA()])
	for i in pbar(idxs):
		data.append(features[i])
		
	return data
	
	
def generate_codebook(features, n_clusters):
	"""Generate visual vocabulary, ie. clusterize 
	extracted features.
	Centroids form the codebook.
	"""
	
	clt = MiniBatchKMeans(n_clusters=n_clusters)
	clt.fit(features)
	
	return clt.cluster_centers_
	
def save_codebook(data_array, path):
	"""Save a pickle file"""
	with open(path, 'wb') as file:
		cPickle.dump(data_array, file)
		
def load_codebook(path):
	"""Load a pickle file"""
	with open(path, 'rb') as file:
		return cPickle.load(file)