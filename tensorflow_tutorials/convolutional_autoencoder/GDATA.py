#Creates 3D data

import os
import numpy as np

def gen_data(input_shape, output_shape, volume):

	''' Input:
	Input_shape: shape of the input image, 3D
	Output_shape: 3D to 5D
	volume: sample 3D image
	
	Output:
	sample: the 5D version of the image
	'''
	
	data_shape = input_shape
	
	sample = np.empty(output_shape)	
	index = np.empty([3], dtype=np.int32)
	for i in range(output_shape[0]):
		index[0] = np.random.randint(data_shape[0]-output_shape[1]+1)
		index[1] = np.random.randint(data_shape[1]-output_shape[2]+1)
		index[2] = np.random.randint(data_shape[2]-output_shape[3]+1)
		sample[i,:,:,:,0] = volume[ index[0]:index[0]+output_shape[1], index[1]:index[1]+output_shape[2], index[2]:index[2]+output_shape[3]]
	
	return sample
	
