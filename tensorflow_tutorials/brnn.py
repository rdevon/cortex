from __future__ import print_function

import numpy as np

import tensorflow as tf
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse



import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

display_step = 10

n_output = 3
n_input = 3


def get_data(datafile, labelfile,seed=21,t=500,delim=',',ty=np.float32,steps=None):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    print('loaded label')
    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data
    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    #all_Y = np.eye(num_labels)[target]  # One liner trick!
    np.random.seed(seed)
    p = np.random.permutation(N)
    data = data[p]
    target = target[p]
    print('returning data')
    if steps == None:
        return data,target
    else:
        return data[:,:steps],target

def BiRNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    print("in birnn " + str(n_hidden))
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print("made left and right")
    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights['out'])# + biases['out']

def make_one_hot_2(target,labels):
    print(target.shape)
    print(labels)
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target-1] = 1
    return targets

def make_one_hot(target):
    targets = (np.arange(target.max()+1) == target[:,:,None]).astype(int)
    return targets
def organize_data(datas,timesteps):
    print(n_input)
    print('hi')
    r = []
    for d in datas:
        subjects = d.shape[0]
        print(d.shape)
        data_size = d.shape[1]
        removal = data_size % timesteps
        data_size = data_size - removal
        step_size = n_input
        print(step_size)
        print(timesteps)
        print(d.shape)
        print(d[:,:data_size].shape)
        r.append(d[:,:data_size,:].reshape(subjects,timesteps,step_size))
    return r


idxs = [[range(0, 495), range(495, 550), range(550, 583)],
        [range(55, 550), range(0, 55), range(550, 583)],
        [range(0, 55) + range(110, 550), range(55, 110), range(550, 583)],
        [range(0, 110) + range(165, 550), range(110, 165), range(550, 583)],
        [range(0, 165) + range(220, 550), range(165, 220), range(550, 583)],
        [range(0, 220) + range(275, 550), range(220, 275), range(550, 583)],
        [range(0, 275) + range(330, 550), range(275, 330), range(550, 583)],
        [range(0, 330) + range(385, 550), range(330, 385), range(550, 583)],
        [range(0, 385) + range(440, 550), range(385, 440), range(550, 583)],
        [range(0, 440) + range(495, 550), range(440, 495), range(550, 583)]
        ]
def run(data_path = "./data_right.txt",labels_path="./labels_right.txt",delim=','):
    global n_steps, n_hidden, n_input, display_step,batch_size,learning_rate,training_iters
    Pool_data, Pool_labels = get_data(data_path,labels_path, delim=delim,ty=np.int32)
    OUTPUT = []
    Pool_data = make_one_hot(Pool_data)
    Pool_labels = make_one_hot_2(Pool_labels,2)
    total_size = Pool_data.shape[1]
    print(total_size)
    print(n_steps)
    n_input = Pool_data.shape[2] *(total_size - (Pool_data.shape[1] % n_steps)) / n_steps
    print(n_input)
    #n_steps = Pool_data.shape[1]
    for i in range(len(idxs)):
        print(n_hidden)
        print(n_steps)
        tf.logging.set_verbosity(tf.logging.ERROR)
        weights = {
            # Hidden layer weights => 2*n_hidden because of forward + backward cells
            'out': tf.Variable(tf.random_normal([n_steps,2 * n_hidden,n_input]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_steps]))
        }
        with tf.device('/gpu:0'):
            x = tf.placeholder("float", [None,n_steps,n_input])
            I = tf.Variable(tf.random_normal([100, 2]))
            y = tf.placeholder("float", [None, 2])
        SNP_label_train = Pool_labels[idxs[i][0]]
        SNP_label_test = Pool_labels[idxs[i][1]]
        SNP_data_train,SNP_data_test = organize_data([Pool_data[idxs[i][0]],Pool_data[idxs[i][1]]],n_steps)
        print('got data and organized')
        pred = BiRNN(x, weights, biases)
        print("break")
        print(SNP_data_test.shape)
        print(SNP_data_train.shape)
        reshape = tf.reshape(pred,[tf.shape(pred)[1],n_steps,n_input])
        flat = tf.reduce_mean(reshape,2)
        #SNP_label_train = make_one_hot(Pool_labels[idxs[i][0]], 2)
        #SNP_label_test = make_one_hot(Pool_labels[idxs[i][1]], 2)
        # Define loss and optimizer
        out = tf.matmul(flat,I)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
        print('got optimizer')
        # Evaluate model
        #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

        predict = tf.argmax(out, axis=1)
        # Initializing the variables
        init = tf.global_variables_initializer()
        fold = []
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            step = 1
            # Keep training until reach max iterations

            for epoch in range(training_iters):
                for i in range(1,len(SNP_data_train),batch_size):
                    batch_x= SNP_data_train[i:i+batch_size]
                    batch_y= SNP_label_train[i:i+batch_size]
                    #batch_x.reshape(n_steps,cur_size,n_input
                    cur_size=batch_x.shape[0]
                    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                # Reshape data to get 28 seq of 28 elements
                #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
                # Run optimization op (backprop)
                if epoch % display_step == 0:
                    # Calculate batch accuracy
                    #SNP_data_train.reshape(n_steps,495,n_input)
                    acc = np.mean(np.argmax(SNP_label_train,axis=1)==
                                  sess.run(predict, feed_dict={x: SNP_data_train, y: SNP_label_train}))
                    # Calculate batch loss
                    #batch_x.reshape(n_steps,cur_size,n_input)
                    #SNP_data_test.reshape(n_steps,55,n_input)
                    #loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print("Iter " + str(epoch)  + ", Training Cost= " + \
                          "{:.5f}".format(acc))
                    test_acc = np.mean(np.argmax(SNP_label_test,axis=1)==
                                       sess.run(predict, feed_dict={x: SNP_data_test, y: SNP_label_test}))
                    print("Testing Cost: " + str(test_acc))


                    fold.append(test_acc)
            print("Optimization Finished!")
            OUTPUT.append(fold)
            # Calculate accuracy for 128 mnist test images
            test_len = SNP_data_test.shape[0]
            test_data = SNP_data_test#mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
            #test_label = SNP_label_test#mnist.test.labels[:test_len]
            #print("Testing Accuracy:", \
            #    sess.run(accuracy, feed_dict={x: test_data, y: test_data}))
        tf.reset_default_graph()
    np.savetxt('output_results_from_BRNN.txt',OUTPUT,delimiter=',')
if __name__ == '__main__':
    global learning_rate, training_iters, batch_size, n_hidden, n_steps
    n_steps = 100
    learning_rate = 0.001
    training_iters = 200
    batch_size = 20
    n_hidden = 10
    args = sys.argv
    if len(args) > 0:
        parse = argparse.ArgumentParser()
        parse.add_argument('-d',action='store',dest='data',help='data file location',default="./data_right.txt")
        parse.add_argument('-l', action='store',dest='labels',help='label file location', default="./labels_right.txt")
        parse.add_argument('-hidden', action='store',dest='h',help='hidden layers', default=n_hidden)
        parse.add_argument('-delta', action='store',dest='d',help='learning rate', default=learning_rate)
        parse.add_argument('-epochs', action='store',dest='e',help='number of epochs', default=training_iters)
        parse.add_argument('-batch', action='store',dest='b',help='batch size', default=batch_size)
        results = parse.parse_args()
        learning_rate = results.d
        training_iters = results.e
        batch_size=results.b
        n_hidden = results.h
        data = results.data
        labels = results.labels
        run(data,labels)
    else:
        run()
