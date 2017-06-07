# Run only this code to test the autoencoder

import os
import numpy as np
import tensorflow as tf
from nibabel.testing import data_path
example_filename = os.path.join(data_path, 'example4d.nii.gz')
import nibabel as nib
import VCAE as AE
import GDATA as GD

def test_nii():

	img = nib.load(example_filename)
	volume = img.get_data()
	volume = volume[:, :, :, 0]
	input_shape = volume.shape
	output_shape = np.expand_dims(volume, axis = 3)
	output_shape = np.expand_dims(output_shape, axis = 3)
	output_shape = output_shape.shape
	
	padding = 'SAME'
	stride = [1,2,2,2,1]
	
	ae = AE.autoencoder(output_shape, padding, stride)
	
	learning_rate = 0.01
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	train = GD.gen_data(input_shape, output_shape, volume)
	n_batches = 10 #10
	batch_size = 100 #100
	n_epochs = 50 # 50
	print "\nepoch" , "\t" , "cost"
	
	for k in range(n_epochs):
		for i in range(n_batches):
			for j in range(batch_size):
				data = GD.gen_data(input_shape, output_shape, volume)
				#train += data
				#mean = train/((k+1) * (i+1) * (j+1))
				#data = data - mean
				sess.run(optimizer, feed_dict = {ae['x']:data})		
		print k, "\t", sess.run(ae['cost'], feed_dict={ae['x']: data})

if __name__ == '__main__':
	test_nii()
