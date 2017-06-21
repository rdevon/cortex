**CNN Embedding**

Key parameters:
 - *input_size* = The original number of features
 - *embedded_size* = The new size of the embedded layer
 - *number_of_filters* = The number of filters with a given size
 - *filter_sizes* = A list of the size of the different filter sizes (in our case, there are 3)
	
An embedded CNN is a Convolutional Neural Network that embeds the input features into a smaller set with the goal of representing the input as a simpler and more condensed form.  Lucky for us, Tensorflow has a quick and simple way to embed features.

`self.embedded_chars = tf.nn.embedding_lookup(weight_tensor, input_tensor)`

Where `weight_tensor` is the embedding weights that connect to the input layer.  The `embedding_lookup` function essentially creates a lookup table between the input and a lower dimension set.  The problem here is that the tensor returned from `embedding_lookup` does not contain the channel dimensions which the `conv2d` function requires.  So, we can use:

`self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)`

Which will expand the dimensions and give us 4 dimensions of [*batch_size*, *input_size*, *embedded_size*, 1].  Where the 1 represents the one channel we have.  

Now that we have the embedded layer, we can use it in our CNN.  In our example, we have 3 *filter_size* in *filter_sizes*.  
For each *filter_size*, we use:

`conv = tf.nn.conv2d(self.embedded_chars_expanded,W,strides=[1, 1, 1, 1],padding="VALID")`

'W' represents our weight set for that filter.  It is of size [*filter_size*, *embedding_size*, 1, *number_of_filters*].    
strides define how many cells the filter moves each time it moves.  The "VALID" padding tells tensorflow to add no padding and drop any cells that are not covered.

As per a usual CNN, we add nonlinearity.  In this case, it is ReLU:

`h = tf.nn.relu(tf.nn.bias_add(conv, b))`

Here, we also add the bias (b) to the layer.  
Next, we use a max pool layer.  The max pool layer down-samples by partitioning the input and taking the max value from each partition.

`pooled = tf.nn.max_pool(h, ksize=[1, input_size - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')`

h is our input layer given by each filter.  The *ksize* argument is the size of the partition.  The other arguments are the same as seen above.

Then, we concatenate the pooled layers (3, in our example case) and reshape them to a 2D matrix of size [1, *number_of_filters***len(filter_sizes)*]

This final pooled layer has whatever applications required to generate predictions.  In our case, we use dropout and feed it into a final output layer of size 2 (as we have 2 classes).  Then, as per a usual Neural Network in Tensorflow, loss functions are built and the model is trained.