import tensorflow as tf
import numpy as np
import math


def autoencoder(input_shape, n_filters, filter_sizes, t):
	
	"""Input Parameters:
	input_shape: shape of the input tensor that will go through the autoencoder.
	n_filters: Number of filters (values indicates n_output of each layer)
	filter_sizes: sizes of each set of filters
	t: input format specifier, number of values per pixel in the input.
	samples:
	mnist: input_shape=[None, 784], n_filters=[10, 10, 10], filter_sizes=[3, 3, 3], t=1
	cifar10: input_shape=[None, 1024], n_filters=[10, 10, 10], filter_sizes=[3, 3, 3], t=1
	
	Output Parameters:
	x: input tensor
	z: output after compression
	y: regenrerated output
	cost: loss after regeneration
	summary_op: tensorboard summary variable for tracking cost	
	"""
	x = tf.placeholder(tf.float32, input_shape, name='x')
	# ensure 2D tensor
	if len(x.get_shape()) == 2:
		x_dim = np.sqrt(x.get_shape().as_list()[1])
		# ensure square image
		if x_dim != int(x_dim):
			raise ValueError('Unsupported image dimensions, supports only square images')
		x_dim = int(x_dim)
		x_tensor = tf.reshape(x, [-1, x_dim, x_dim, t])
	elif len(x.get_shape()) == 4:
		x_tensor = x
	else:
		raise ValueError('Unsupported input dimensions')
	current_input = x_tensor
	encoder = []
	shapes = []
	#create the structure
	for layer_i, n_output in enumerate(n_filters):  
		n_input = current_input.get_shape().as_list()[3]
		shapes.append(current_input.get_shape().as_list())
		W = tf.Variable(tf.random_uniform([filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
		b = tf.Variable(tf.zeros([n_output]))
		encoder.append(W)                  
		x1 = tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b)
		f1 = 0.5 * (1 + 0.2)
		f2 = 0.5 * (1 - 0.2)
		output = f1 * x1 + f2 * abs(x1)
		current_input = output
	
	z = current_input #output after phase 1, compression
	
	encoder.reverse() #start reverse process, regeneration
	shapes.reverse()
	
	for layer_i, shape in enumerate(shapes):
		W = encoder[layer_i]
		b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
		x2 = tf.add(tf.nn.conv2d_transpose(current_input, W, tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]), strides=[1, 2, 2, 1], padding='SAME'), b)
		f1 = 0.5 * (1 + 0.2)
		f2 = 0.5 * (1 - 0.2)
		output = f1 * x2 + f2 * abs(x2)
		current_input = output
		
	y = current_input # regenerated inputs after phase 2
	
	with tf.name_scope('cost'):  # track cost for tensorboard
		cost = tf.reduce_sum(tf.square(y - x_tensor))
		
	tf.summary.scalar("cost", cost)
	summary_op = tf.summary.merge_all()
	
	return {'x': x, 'z': z, 'y': y, 'cost': cost, 'summary_op': summary_op}
