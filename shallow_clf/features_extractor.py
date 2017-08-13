import numpy as np
import cv2

 
class RootSIFT(object):
	"""Extension to SIFT as described in:http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.370.7498&rep=rep1&type=pdf
	Computing Euclidean distance in the feature map space is 
	equivalent to Hellinger distance in the original space. 
	Hellinger distance is more suitable for comparing histograms
	and less sensitive to high bin values. 
	"""
	
	
	def __init__(self):
		
		self.detector = cv2.FeatureDetector_create("GFTT")
		self.extractor = cv2.DescriptorExtractor_create("SIFT")
			
 
	def compute(self, image, eps=1e-7):
		
		#Extract keypoints features 
		kps = self.detector.detect(image)
		
		if len(kps) == 0:
			return []
		
		_, features = self.extractor.compute(image,kps)
		
 
		#L1 normalize features
		features /= (features.sum(axis=1, keepdims=True) + eps)
		
		#Root square normalized features
		features = np.sqrt(features)
 
		return features
		
"""
TO-DO: Add more descriptors, change the implementation
that feature descriptor and extractor can be selected
using a configuration file.
"""