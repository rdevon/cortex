MLP.py
    A generalized file that will build an MLP based on given input
If it is run as a main, it will run an MLP using MNIST and a hardcoded set
of parameters to show correctness
'construct_automatically' is the primary entry point.

BdRNN.py
    This file will build a bi-directional RNN with given inputs.  If run
as main, it will use the MNIST dataset to output for each timestep and this
output is compared to the inputs.
    The function 'construct_automatically' will return output for each timestep
as well as the output state from each single RNN.  The 'output[-1]' is the final
output from the BdRNN, which can be used to classify.

multimodal.py
    This is an example test for two MLPs using two datasets and an open code
repo to parse the data.  The data is found here:
    http://multimedia-commons.s3-website-us-west-2.amazonaws.com/?prefix=subsets/YLI-MED/
     - subsets/YLI-MED/features/audio/mfcc20/mfcc20.tgz
     - subsets/YLI-MED/features/keyframe/alexnet/fc7.tgz
    Place these datasets into the img_audio_data folder


    And the code for parsing is found here:
    https://github.com/lheadjh/MultimodalDeepLearning
        - The img_audio_data/mdl_data.py is the file that does most of the work
        to parse the data

