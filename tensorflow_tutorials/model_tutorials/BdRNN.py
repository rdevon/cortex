import tensorflow as tf
import numpy as np
from utilities import *
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#For the scripted example, we use the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)



'''This builds the bidirectional RNN.  It only builds one layer of the RNN.
For a deeper model, stack BdRNNs together
    The basic idea is that we build two RNNs (left and right), and combine them
into a bdRNN.
    This function returns the outputs (the two RNNs concatenated) and the two states.
For classification purposes, the final state (outputs[-1] can be used as the predictor)

args: 
    x=The input tensor
    weight=A single weight set
    bias=a single bias set.  Set to zero if this is the final output
    
returns:
    outputs=Final output states of all timesteps
    SL=The final timestep output for the left side
    SR=The final timestep output for the right side
'''
def BdRNN(x, weight, bias, n_steps, n_hidden):

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    try:
        outputs, SL, SR = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: #The older versions of TF do not produce the two output states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    return tf.matmul(outputs, weight) + bias, SL, SR


'''This function creates a single bdRNN.
args:
    n_steps=The number of timesteps
    n_hidden=The number of hidden nodes
    n_input=The size of the input
    
returns:
    X=The input tensor
    y=The label tensor
    pred=The output from the bdRNN

'''
def construct_automatically(n_steps, n_hidden, n_input, **kwargs):
    X = tf.placeholder("float", [None, n_steps, n_input], name='input_variable')
    y = tf.placeholder("float", [n_steps, None, n_input], name='labels_variable')

    weights = tf.Variable(tf.random_normal([n_steps, 2 * n_hidden, n_input]),name='RNN_weights_'+str(n_steps))

    #The biases are optional.  Mainly because this first layer can be the output
    #layer
    if 'bias' in kwargs:
        if kwargs['bias']:
            bias = tf.Variable(tf.random_normal([n_input, n_steps]))
            pred, _, _ = BdRNN(X, weights, bias)
    else:
        pred, _, _ = BdRNN(X, weights, 0,n_steps, n_hidden)
    return X, y, pred


def launch_mnist(x, y, optimizer,cost,init=tf.global_variables_initializer(),training_iters=100,batch_size=20,display_step=10):
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)

        for epoch in range(training_iters):
                batch_x, _ = mnist.train.next_batch(batch_size)

                #The MNIST data must be reshaped to fit the RNN (timesteps + input size)
                batch_x = batch_x.reshape(batch_size,n_steps,n_input)
                batch_y = batch_x.reshape(n_steps,batch_size,n_input)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                if (epoch) % display_step == 0:
                    # Calculate batch accuracy
                    acc = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(epoch) + ", Training cost= " + \
                          "{:.5f}".format(acc))
        test_length = 128
        test_data = mnist.test.images[:test_length].reshape(test_length, n_steps, n_input)
        test_data_out = test_data.reshape((n_steps, test_length, n_input))
        print("Testing cost:", \
              sess.run(cost, feed_dict={x: test_data, y: test_data_out}))
    print("Optimization Finished!")


if __name__ == '__main__':
    training_iters = 500
    batch_size = 25
    display_step = 5

    n_input = 28
    n_steps = 28
    n_hidden = 128

    X, y, pred = construct_automatically(n_steps, n_hidden, n_input)

    cost = tf.reduce_mean(tf.square(y - pred))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    init = tf.global_variables_initializer()

    launch_mnist(X, y, optimizer, cost, init, training_iters, batch_size, display_step)