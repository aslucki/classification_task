from sklearn.metrics import pairwise
import numpy as np

def describe(features, codebook):
	"""Create histogram of visual words occurance in the features set"""
	
	#Calculate discance between every feature and word in the codebook
	distances = pairwise.euclidean_distances(features, Y=codebook)
	
	#Count words occurances
	(words, counts) = np.unique(np.argmin(distances, axis=1), return_counts=True)
	hist = np.zeros((len(codebook),), dtype="float")
	hist[words] = counts
	
	return hist