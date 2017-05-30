# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn>=0.18.1, tensorflow>=1.0

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
import sys
import argparse

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    biases = tf.random_normal([shape[1]])
    return tf.Variable(weights), tf.Variable(biases)

def forwardprop(X, weights,biases,dropout=None):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    #output = weights.pop(-1)
    h = tf.nn.sigmoid(tf.add(tf.matmul(X,weights.pop(0)),biases.pop(0)))
    print "forward " + str(len(weights))
    for w in range(len(weights)):
        h = tf.nn.sigmoid(tf.add(tf.matmul(h,weights[w]),biases[w]))
        if dropout != None:
            h = tf.nn.dropout(h,dropout)
     # The \varphi function
    return h
def make_one_hot(target,labels):
    print target.shape
    print labels
    targets = np.zeros((len(target),labels))
    targets[np.arange(len(target)),target-1] = 1
    return targets

def get_data(datafile, labelfile,seed=21,t=500,delim=',',ty=np.float32):
    print('entered get data')
    data   = np.loadtxt(datafile,delimiter=delim,dtype=np.float32)
    print('loaded data')
    print data.shape
    target = np.loadtxt(labelfile,delimiter=delim,dtype=np.int32)
    print('loaded label')
    print target.shape
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
    train_d = data[:t]
    test_d = data[t:]
    #targets = make_one_hot(target,num_labels)
    targets = target
    train_t = targets[:t]
    test_t = targets[t:]
    print('returning data')
    print train_t.shape
    print train_d.shape
    return data, target
def return_weights(input_size,weight_sizes):
    X = tf.placeholder("float", shape=[None, input_size])
    #y = tf.placeholder("float", shape=[None, output_size])
    weights = []
    biases = []
    in_layer_size = input_size
    for size in weight_sizes:
        w,b=init_weights((in_layer_size, size))
        weights.append(w)
        biases.append(b)
        in_layer_size=size
    print len(weights)
    return(X,weights,biases)
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
def main():
    pool_MRI_X_train, pool_MRI_y_train = get_data('./data.csv', './labels.csv')
    pool_SNP_X_train, pool_SNP_y_train = get_data('./data_right.txt', './labels_right.txt', delim=',',
                                                                ty=np.int32)

    for i in range(len(idxs)):
        train_MRI_X, test_MRI_X = pool_MRI_X_train[idxs[i][0]],pool_MRI_X_train[idxs[i][1]]

        train_MRI_y, test_MRI_y = pool_MRI_y_train[idxs[i][0]],pool_MRI_y_train[idxs[i][1]]
        train_SNP_X, test_SNP_X, =pool_SNP_X_train[idxs[i][0]],pool_SNP_X_train[idxs[i][1]]
        train_SNP_y, test_SNP_y = pool_SNP_y_train[idxs[i][0]],pool_SNP_y_train[idxs[i][1]]
        print train_SNP_X.shape
        #y_MRI_size = train_MRI_y.shape[1]
        x_MRI_size = train_MRI_X.shape[1]
        h_MRI_1_size = 6000
        h_MRI_2_size = 500
        h_MRI_3_size = 200
        y_MRI_size = train_MRI_y.shape[1]
        #print train_MRI_y.shape
        #print train_MRI_X.shape
        print "break"
        print train_SNP_y.shape
        print train_SNP_X.shape
        train_SNP_y = make_one_hot(train_SNP_y,2)
        test_SNP_y = make_one_hot(test_SNP_y,2)
        x_SNP_size = train_SNP_X.shape[1]
        h_SNP_1_size = 3000
        h_SNP_2_size = 500
        #h_SNP_3_size = 500
        h_SNP_4_size = 200
        y_SNP_size = train_SNP_y.shape[1]

        h_output_size = 400
        y_output_size = train_SNP_y.shape[1]
        print "here"
        print y_output_size
        print x_SNP_size
        combined_layer_size = 400
        # Symbols
        #with tf.device("/gpu:0"):
        X,weights,biases = return_weights(x_MRI_size,[h_MRI_1_size,h_MRI_2_size,h_MRI_3_size])
        #,h_SNP_3_size
        X_SNP,weights_SNP,biases_SNP = return_weights(x_SNP_size,[h_SNP_1_size, h_SNP_2_size,h_SNP_4_size])
        #X = tf.placeholder("float", shape=[None, x_MRI_size])
        y = tf.Variable(tf.random_normal([400, y_output_size]))

        Y = tf.placeholder("float", [None, 2])
        #ow = tf.random_normal(h_output_size, stddev=0.1)
        #out_weights = tf.Variable(ow)
        #ow_ou = tf.random_normal(y_output_size, stddev=0.1)
        #y_out = tf.Variable(ow_ou)
        #out_weights = init_weights(h_output_size,y_output_size)
        # Weight initializations
        #w_1 = init_weights((x_MRI_size, h_MRI_1_size))
        #w_2 = init_weights((h_MRI_1_size,h_MRI_2_size))
        #w_3 = init_weights((h_MRI_2_size, y_MRI_size))
        name = "/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_" + str(1000) + "_" + str(
            300) + "_" + str(1000) + "/model.ckpt.meta"

        # Forward propagation
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        #saver = tf.train.import_meta_graph(name)
        #saver.restore(sess,
        #              tf.train.latest_checkpoint("/home/lhd/tensorflow/For_Tensorflow_Multimodal/model_1000_300_1000/"))
        #graph = tf.get_default_graph()
        #weights_SNP[0] = graph.get_tensor_by_name("W1:0")
        #weights_SNP[1] = graph.get_tensor_by_name("W2:0")
        out_MRI    = forwardprop(X, weights,biases,dropout=.3)
        out_SNP = forwardprop(X_SNP,weights_SNP,biases_SNP,dropout=.5)
        concat = tf.concat([out_MRI,out_SNP],axis=1)
        yhat = tf.matmul(concat,y)

        predict = tf.argmax(yhat, axis=1)

        # Backward propagation
        cost = tf.square(tf.reduce_mean((Y-yhat)))
        updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        # Run SGD
        init = tf.global_variables_initializer()
        sess.run(init)
        batch_size = 20
        for epoch in range(training_iters):
            # Train with each example
            for i in range(1,len(train_MRI_X),batch_size):
                sess.run(updates, feed_dict={X: train_MRI_X[i: i + batch_size],X_SNP:train_SNP_X[i:i+batch_size], Y: train_MRI_y[i: i + batch_size]})

            train_accuracy = np.mean(np.argmax(train_MRI_y, axis=1) ==
                                     sess.run(updates, feed_dict={X: train_MRI_X,X_SNP:train_SNP_X, Y: train_MRI_y}))
            test_accuracy  = np.mean(np.argmax(test_MRI_y, axis=1) ==
                                     sess.run(updates, feed_dict={X: test_MRI_X, X_SNP:test_SNP_X, Y: test_MRI_y}))

            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

        sess.close()

if __name__ == '__main__':
    global learning_rate, training_iters, batch_size, n_hidden, n_steps
    n_steps = 100
    learning_rate = 0.001
    training_iters = 1000
    batch_size = 20
    n_hidden = 1
    args = sys.argv
    if len(args) > 0:
        parse = argparse.ArgumentParser()
        parse.add_argument('-hidden', action='store',dest='h',help='hidden layers', default=n_hidden)
        parse.add_argument('-delta', action='store',dest='d',help='learning rate', default=learning_rate)
        parse.add_argument('-epochs', action='store',dest='e',help='number of epochs', default=training_iters)
        parse.add_argument('-batch', action='store',dest='b',help='batch size', default=batch_size)
        results = parse.parse_args()
        learning_rate = results.d
        training_iters = results.e
        batch_size=results.b
        n_hidden = results.h
        main()
    else:
        main()