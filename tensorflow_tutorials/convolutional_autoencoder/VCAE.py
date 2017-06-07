import os
import numpy as np
import tensorflow as tf
import math

def autoencoder(input_shape, padding, stride):
	
	'''
	input:
	input_shape: the 5D shape of the generated samples
	Padding: type of padding, SAME by default
	stride: Stride, 5D
	
	Output:
	x: input tensor
	z: output after compression
	y: regenrerated output
	cost: loss after regeneration
	'''
	x = tf.placeholder(tf.float32, input_shape)
	
	if len(x.get_shape()) == 5:
		x_tensor = x
	else:
		raise ValueError('Unsupported input dimensions')
		
	curr = x_tensor
	n_input = curr.get_shape().as_list()[3]
	enc_W = tf.Variable(tf.random_uniform([3, 3, 3, n_input, 10], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
	curr = tf.nn.conv3d(curr, enc_W, strides=stride, padding=padding)
	b_shape = curr.get_shape().as_list()[1:]
	enc_b = tf.Variable(tf.fill(b_shape, 0.01), name="enc_b")
	curr += enc_b
	f1 = 0.5 * (1 + 0.2)
	f2 = 0.5 * (1 - 0.2)
	output = f1 * curr + f2 * abs(curr)
	curr = output
	
	z = curr
	
	n_input = curr.get_shape().as_list()[3]
	dec_W = tf.Variable(tf.random_uniform([3, 3, 3, n_input, 10], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
	
	curr = tf.nn.conv3d_transpose(curr, dec_W, output_shape=input_shape, strides=stride, padding=padding)
	b_shape = curr.get_shape().as_list()[1:]
	dec_b = tf.Variable(tf.fill(b_shape, 0.01), name="dec_b")
	curr += dec_b
	f1 = 0.5 * (1 + 0.2)
	f2 = 0.5 * (1 - 0.2)
	output = f1 * curr + f2 * abs(curr)
	curr = output

	
	y = curr
	
	cost = tf.reduce_sum(tf.square(y - x_tensor))
	
	return {'x': x, 'z': z, 'y': y, 'cost': cost}
