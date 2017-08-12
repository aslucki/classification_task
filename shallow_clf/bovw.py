from sklearn.metrics import pairwise
import numpy as np

def describe(features, codebook):
	distances = pairwise.euclidean_distances(features, Y=codebook)
	(words, counts) = np.unique(np.argmin(distances, axis=1), return_counts=True)
	hist = np.zeros((len(codebook),), dtype="float")
	hist[words] = counts
	
	return hist
	
	
def get_batches(features, batch_size=1000):
	start = 0
	while True:
		batch = features[start:start + batch_size]
		if len(batch) == 0: break
		
		yield batch
		
		if len(batch) < batch_size: break
		start += batch_size