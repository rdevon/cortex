from __future__ import print_function

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

'''This file creates an MLP model with given parameters.  The functions can be
imported to be used at your leisure.  Or, if the file is run as a main, it builds
an MLP and uses the MNIST dataset to test the model
'''


'''This returns the model.  It's the set of all functions for the forward pass
If you already have the input variable and the weights or biases, this function
can be called manually
args:
    x=Input variable
    weights=A weight set for the entire model
    biases=The biases for the model
    dropout=A scalar representing the dropout percentage (optional)
output:
    The final output after all forward pass functions
To note:  matmul only works on 2D tensors
'''
def multilayer_perceptron(x, weights, biases, dropout=None):
    for weight,bias in zip(weights,biases):
        x = tf.add(tf.matmul(x, weight), bias)
        if dropout != None:
            x = tf.nn.dropout(x, dropout)
    return x


'''This will construct the weights and biases automatically.  As well as the input
and output tensors.  Then, it builds the functions through the multilayer_perceptron
function
args:
    input_size=The size of the input
    layer_sizes=list of layer sizes (including output size)
    dropout=A single value representing percentage
    X=An optional place to add your own input tensor
output:
    X=The input variable
    y=the label variable
    pred=Model output, or prediction variable
'''
def construct_automatically(input_size,layer_sizes, dropout=None, X=None):
    weights = []
    biases = []

    #These are the input and output tensors
    if X==None:
        X = tf.placeholder("float", [None, input_size])
    else:
        x = X
    y = tf.placeholder("float", [None, layer_sizes[-1]])

    #These are the weights and biases for the hidden layers
    for i,layer in enumerate(layer_sizes):
        weights.append(tf.Variable(tf.random_normal([input_size,layer])))
        biases.append(tf.Variable(tf.random_normal([layer])))
        input_size = layer


    #Here we build the functions
    pred = multilayer_perceptron(X,weights,biases,dropout)
    return X, y, pred


'''This will create a session and run it using the MNIST dataset
NOTES:
    The session is what starts and guides the computation.
Before this step, the graph has already been built and consists of all of the
operations you previously made.  But, when Session.run() is called, it places 
this graph on the GPU.  The 'init' variable is an initializer op that 
initializes all of the variables in the graph.

    The subsequent sess.run() calls are used for specific operations.  In this
case, by using the optimizer and cost functions, we're calling every op 
from that part of the graph.  So, it can be used for a single operation, such
as a single matmul, or, as it is in this case, the forward pass for the model
as well as the backpropagation.
The docs for this are found here:
    https://www.tensorflow.org/api_guides/python/train
All optimizers use the minimize() function to minimize some error function.  
This minimize function also calls compute_gradients() and apply_gradients().

    In this example, sess.run(...) returns the output from calls to both the
optimizer and cost.  Only the output from the cost function is printed.  
'''
def launch_mnist(x, y, optimizer,cost,init=tf.global_variables_initializer(),training_epochs=100,batch_size=20,display_step=10):
    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    # Parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of features
    n_hidden_2 = 256  # 2nd layer number of features
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)

    X, y, pred = construct_automatically(n_input,[n_hidden_1,n_hidden_2,n_classes])
    #pred = multilayer_perceptron(x, w, b)

    # Define loss and optimizer
    #The cost function can be anything you desire
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the global variables (this gives a default graph)
    init = tf.global_variables_initializer()
    launch_mnist(X, y, optimizer, cost, init, training_epochs, batch_size, display_step)
