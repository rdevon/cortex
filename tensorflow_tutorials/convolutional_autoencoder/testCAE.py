# testing the 2D CAE using mnist dataset, run this file and keep CAE.py in the same directory as this file.


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import math
import matplotlib
matplotlib.use('pdf');
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf1
import CAE as AE

tf.reset_default_graph()

logs_path = "1/"

def test_mnist():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)        
    mean_img = np.mean(mnist.train.images, axis=0)
    
    ae = AE.autoencoder(input_shape=[None, 784], n_filters=[10, 10, 10], filter_sizes=[3, 3, 3], t=1)
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    batch_size = 100
    n_epochs = 50
    	
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    	
    print "\nepoch" , "\t" , "cost"
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            _, summary = sess.run([optimizer, ae['summary_op']], feed_dict={ae['x']: train})
            writer.add_summary(summary, epoch_i)
            
        print epoch_i, "\t", sess.run(ae['cost'], feed_dict={ae['x']: train})

    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    
    with pdf1.PdfPages('testImages.pdf') as pdf:
        for example_i in range(n_examples):
            plt.imshow( np.reshape(test_xs[example_i, :], (28, 28)))
            fig1 = plt.draw()
            pdf.savefig(fig1)
       
            plt.imshow(np.reshape(np.reshape(recon[example_i, ...], (784,)) + mean_img,(28, 28)))
            fig2 = plt.draw()
            pdf.savefig(fig2)
   
    print "\nFind images and reconstruction in the file testImages.pdf in the home directory\n"
       
if __name__ == '__main__':
	test_mnist()
