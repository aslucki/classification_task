"""
Extract CNN codes from selected layer
of VGG16 model.

Dense layers codes can't be extracted
from images of dimensions other than (224,224).

Because of pooling layers, any of input	images
dimension can't be less than 48.
"""

from __future__ import print_function
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
import cv2

class VGG16_layer(object):
	
	def __init__(self,layer_name, image_size=(224,224)):
		self.image_size = image_size
		
		#Check if image size other than (224,224)
		if image_size != (224,224):
			if any(dim < 48 for dim in image_size):
				raise ValueError('Dimensions are too small for this CNN, min: (48,48)')
			else:
				#Exclude top (dense) layers
				self.model = VGG16(weights='imagenet', include_top=False)
		else:
			#Load whole model with weights
			base_model = VGG16(weights='imagenet')
			#Leave structure up to selected layer
			self.model = Model(inputs=base_model.input, 
						  outputs=base_model.get_layer(layer_name).output)
			
	
	def compute(self, image_array, batch_size=32):
		#Check if passed image should be resized
		if image_array.shape[1:3] != self.image_size:
			resized = []
			for image in image_array:
				resized.append(cv2.resize(image, self.image_size[::-1]))
			image_array = np.asarray(resized)
		
		#Change data type to perform next operation
		image_array = image_array.astype('float')
		
		#Zero-center channels with means from imagenet dataset
		image_array[:,0,:,:] -= 103.939
		image_array[:,1,:,:] -= 116.779
		image_array[:,2,:,:] -= 123.68
		
		#Extract CNN codes from layer
		features = self.model.predict(image_array, batch_size=32)
		
		return features
		


	