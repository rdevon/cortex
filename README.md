# cortex2.0
A machine learning library for PyTorch


## SETUP
The first thing to do is to set up the config.yaml. This file is user-specific (it got added at some point, so I need to fix this), and will tell cortex everything user-specific regarding data locations, visualation, and outputs. Here is mine on MILA machines:

`code()`
torchvision_data_path: /data/milatmp1/hjelmdev/data/
data_paths: {
    Imagenet-12: /data/lisa/data/ImageNet2012_jpeg,
    CelebA: /tmp/hjelmdev/CelebA}
viz: {
    font: /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf,
    server: 'http://132.204.26.180'
}
out_path: /data/milatmp1/hjelmdev/outs/

These are as follows:

* torchvision_data_path: the path to all torchvision-specific datasets (details can be found in torchvision.datasets)
* data_paths: user-specified custom datasets. Currently, only support is for image folders (a la imagenet), but other dataset types (e.g., text) are planned in the near-future.
* vis: visdom specific arguments.
* out_path: Out path for experiment outputs

To run a simple experiment, try  `python main.py classifier -S MNIST -n test_classifier`

This should run a simple classifier on MNIST.
