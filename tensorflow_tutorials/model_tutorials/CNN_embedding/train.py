#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
from text_cnn import TextCNN
from tensorflow.contrib import learn
from utilities import *


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 500, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 200, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 20, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

training_size = 3000
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
#These are example datasets of SNPs
data_path_SNP = "./data_sample.txt"
labels_path_SNP = "./labels_sample.txt"
x = get_data(data_path_SNP, delim=',')
y = make_one_hot(get_data(labels_path_SNP, type=np.int32, delim=','), 2)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Training
# ==================================================

with tf.Graph().as_default():
	print "training"
	x_train, x_dev = x_shuffled[:15], x_shuffled[15:]
	y_train, y_dev = y_shuffled[:15], y_shuffled[15:]

	batches = list(zip(x_train, y_train))
	batches = [batches[i:i + FLAGS.batch_size] for i in range(0, len(batches), FLAGS.batch_size)]

	session_conf = tf.ConfigProto(
	  allow_soft_placement=FLAGS.allow_soft_placement,
	  log_device_placement=FLAGS.log_device_placement)
	sess = tf.Session(config=session_conf)
	with sess.as_default():
		cnn = TextCNN(
			sequence_length=x.shape[1],
			num_classes=y.shape[1],
			vocab_size=3,
			embedding_size=FLAGS.embedding_dim,
			filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
			num_filters=FLAGS.num_filters,
			l2_reg_lambda=FLAGS.l2_reg_lambda)

		# Define Training procedure
		global_step = tf.Variable(0, name="global_step", trainable=False)
		optimizer = tf.train.AdamOptimizer(1e-3)
		grads_and_vars = optimizer.compute_gradients(cnn.loss)
		train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

		sess.run(tf.global_variables_initializer())

		def train_step(x_batch, y_batch):
			"""
			A single training step
			"""
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
			}
			_, step, accuracy = sess.run(
				[train_op, global_step, cnn.accuracy],
				feed_dict)
			time_str = datetime.datetime.now().isoformat()

		def dev_step(x_batch, y_batch):
			"""
			Evaluates model on a dev set
			"""
			feed_dict = {
			  cnn.input_x: x_batch,
			  cnn.input_y: y_batch,
			  cnn.dropout_keep_prob: 1.0
			}
			accuracy = sess.run(
				cnn.accuracy,
				feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print " testing accuracy: " + str(accuracy)

		step = 0

		for j in range(training_size):
				# Training loop. For each batch...
				#print "training batches"

				for batch in batches:
					#print step
					x_batch, y_batch = zip(*batch)
					train_step(x_batch, y_batch)
					current_step = tf.train.global_step(sess, global_step)
				if j % 10 == 0:
					print("\nstep: "+str(j)+"\nEvaluation:")
					dev_step(x_dev, y_dev)
					#print("training accuracies: ")
					#dev_step(x_train, y_train)
					np.random.shuffle(batches)
					print len(batches)
					print("")