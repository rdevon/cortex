import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import VCAE as AE

data_path = 'data/'
n_subs = 5
n_cons = 3
n_epochs = 10

def test_nii():
	volumes = {}
	for sub_i in range(n_subs):
		for con_j in range(n_cons):
			example_filename = 's' + str(sub_i) + 'c' + str(con_j) + '.nii'
			example_path = os.path.join(data_path, example_filename)
			img = nib.load(example_path)
			volume = img.get_data()
			shape = img.get_shape()
			volume = np.reshape(volume, [-1, shape[0], shape[1], shape[2], shape[3]])
			shape = volume.shape
			volumes[example_filename] = volume
	
	padding = 'SAME'
	stride = [1,2,2,2,1]
	
	ae = AE.autoencoder(shape, padding, stride)
	
	learning_rate = 0.01
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	print "\nepoch" , "\t" , "cost"
	
	for k in range(n_epochs):
		for i in range(n_subs):
			for j in range(n_cons):
				s = 's' + str(i) + 'c' + str(j) + '.nii'
				data = volumes[s]
				sess.run(optimizer, feed_dict = {ae['x']:data})		
		print k, "\t", sess.run(ae['cost'], feed_dict={ae['x']: data})

if __name__ == '__main__':
	test_nii()
