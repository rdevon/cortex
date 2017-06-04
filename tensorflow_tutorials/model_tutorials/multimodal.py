import tensorflow as tf
import numpy as np
from utilities import *
import BdRNN
import MLP

from img_audio_data import mdl_data
from tensorflow.contrib import rnn

'''NOTES:  This uses two datasets from:
    http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=subsets/YLI-MED/
     - subsets/YLI-MED/features/audio/mfcc20/mfcc20.tgz
     - subsets/YLI-MED/features/keyframe/alexnet/fc7.tgz
The code to parse this information is from: 
    https://github.com/lheadjh/MultimodalDeepLearning
        - The img_audio_data/mdl_data.py is the file that does most of the work
        to parse the data
'''

'''This was taken from the aforementioned code base.  But, it's a simple make_one_hot
function
'''
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    print num_labels
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + np.ravel(labels_dense)] = 1
    return labels_one_hot

if __name__ == '__main__':
    #The training and test data are parsed from mdl_data, made hot, and then
    #randomized
    data = mdl_data.YLIMED('./YLIMED_info.csv', "./img_audio_data/mfcc20",   "./img_audio_data/fc7")
    X_img_train = data.get_img_X_train()
    X_aud_train = data.get_aud_X_train()
    y_train = data.get_y_train()
    Y_train = dense_to_one_hot(y_train)

    p = np.random.permutation(len(Y_train))
    X_img_train = X_img_train[p]
    X_aud_train = X_aud_train[p]
    Y_train = Y_train[p]

    X_img_test = data.get_img_X_test()
    X_aud_test = data.get_aud_X_test()
    y_test = data.get_y_test()
    Y_test = dense_to_one_hot(y_test)

    #Our parameters
    #Although the input parameters are explicit here, they can be easily inferred
    #from the numpy arrays.
    learning_rate = 0.001
    training_epochs = 100
    batch_size = 256
    display_step = 1

    n_input_img = 4096  # YLI_MED image data input (data shape: 4096, fc7 layer output)
    n_hidden_1_img = 1000  # 1st layer num features 1000
    n_hidden_2_img = 600  # 2nd layer num features 600

    n_input_aud = 2000  # YLI_MED audio data input (data shape: 2000, mfcc output)
    n_hidden_1_aud = 1000  # 1st layer num features 1000
    n_hidden_2_aud = 600  # 2nd layer num features 600

    n_hidden_1_in = 600
    n_hidden_1_out = 256
    n_hidden_2_out = 128

    n_classes = 10  # YLI_MED total classes (0-9 digits)
    dropout = 0.75

    #Here, we create two models.  One for each input 'sleeve'
    #then, they are added together (another option is to concatenate them)
    #That output is then sent through a combined model, whose output is used
    #to classify
    X_img, y_img, output_img = MLP.construct_automatically(n_input_img,[n_hidden_1_img,n_hidden_2_img])

    X_aud, y_aud, output_aud = MLP.construct_automatically(n_input_aud,[n_hidden_1_aud,n_hidden_2_aud], dropout=dropout)

    added = tf.add(output_img, output_aud)

    _, y_comb, output_comb = MLP.construct_automatically(n_hidden_1_in, [n_hidden_1_out, n_hidden_2_out, n_classes],X=added)

    #The cost and optimizer functions
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_comb, labels=y_comb))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(len(Y_train)/batch_size)
            for i in range(total_batch):
                batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_train, X_img_train, Y_train, batch_size, len(Y_train))

                sess.run(optimizer, feed_dict = {X_aud: batch_x_aud, X_img: batch_x_img, y_comb: batch_ys})

                avg_cost += sess.run(cost, feed_dict = {X_aud: batch_x_aud, X_img: batch_x_img, y_comb: batch_ys}) / total_batch

                if finish:
                    p = np.random.permutation(len(Y_train))
                    X_aud_train = X_aud_train[p]
                    X_img_train = X_img_train[p]
                    Y_train = Y_train[p]

            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
        print "Optimization Finished!"

        # Test the final model
        #correct_prediction = tf.equal(tf.argmax(output_comb, 1), tf.argmax(y_comb, 1))

        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print "Accuracy:", accuracy.eval({X_aud: X_aud_test, X_img: X_img_test, y_comb: Y_test})

