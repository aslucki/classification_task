from __future__ import print_function
from sklearn.cluster import MiniBatchKMeans
from progressbar import Percentage, ProgressBar,Bar,ETA
import numpy as np
import cPickle


def get_random_subset(features, sample_percent):
	features_size = np.asarray(features).shape[0]
	n_samples = int(np.ceil(sample_percent * features_size))
	
	
	idxs = np.random.choice(np.arange(0, features_size), (n_samples), replace=False)
	idxs.sort()
	
	data = []
	
	print('[INFO] Retrieving {} random samples from dataset'.format(n_samples))
	pbar = ProgressBar(widgets=[Bar(), ' ', Percentage(), ' ', ETA()])
	for i in pbar(idxs):
		data.append(features[i])
		
	return data
	
	
def generate_codebook(features, n_clusters):
	clt = MiniBatchKMeans(n_clusters=n_clusters, verbose=True)
	clt.fit(features)
	
	return clt.cluster_centers_
	
def save_codebook(data_array, path):
	with open(path, 'wb') as file:
		cPickle.dump(data_array, file)
		
def load_codebook(path):
	with open(path, 'rb') as file:
		return cPickle.load(file)