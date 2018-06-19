# Warning
Cortex is under heavy development, and should not be used (and issues should not be raised) unless you know what you are doing (i.e., you are a developer for this project).  
# Cortex2.0  
[![Build Status](https://travis-ci.com/joeljpoulin/cortex2.0.svg?branch=setup-pytest)](https://travis-ci.com/joeljpoulin/cortex2.0)  
  
A machine learning library for PyTorch.
    
## Installation

### Prerequisites 
##### Visdom 

    $pip install visdom

### From Source

    $git clone https://github.com/rdevon/cortex2.0.git
    $cd cortex2.0
    $pip install .
    
### Configuration

#### Visdom Server
Start a Visdom server and look for server address in the output. By default, the server's address is `http://localhost:8097`.


    $python -m visdom.server

#### Experiment Configuration

The first thing to do is to set up the config.yaml. This file is user-specific (it got tracked at some point, so I need to fix this), and will tell cortex everything user-specific regarding data locations, visualation, and outputs. 
  
```
$rm -rf ~/.cortex.yml
$cortex setup
```





##### Configuration File Example
Located at `~/.cortex.yml`
```python  
torchvision_data_path: /data/milatmp1/hjelmdev/data/  
data_paths: {  
 Imagenet-12: /data/lisa/data/ImageNet2012_jpeg, CelebA: /tmp/hjelmdev/CelebA}viz: {  
 font: /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf, server: 'http://132.204.26.180'}  
out_path: /data/milatmp1/hjelmdev/outs/  
```  
These are as follows:  

* torchvision_data_path: the path to all torchvision-specific datasets (details can be found in torchvision.datasets)  
* data_paths: user-specified custom datasets. Currently, only support is for image folders (a la imagenet), but other dataset types (e.g., text) are planned in the near-future.  
* vis: visdom specific arguments.  
* out_path: Out path for experiment outputs  
### Usage
##### Help

    cortex --help
##### Options
  
There are many command-line options in cortex:  
  
* `-n` Name of experiment. Experiment is saved in <out_path>/<name>  
* `-S` data Source (from torchvision or user-specified in config.yaml)  
* `-r` Reload model (takes the .t7 file as the argument)  
* `-a` Arguments. This is a `,`-delimited string. For instance, to increase the training epochs on the above example and use sgd, this should work: `python main.py classifier -S MNIST -n test_classifier -a train.epochs=5000,optimizer.optimizer=sgd`  
* `-c` Config (For yamls of arguments, doesn't appear to be working right now)  
* `-k` Klean an experiment directory  
* `-v` Verbosity of logger  
* `-o` Out path (overrides `config.yaml`)  
* `-t` Test mode (for evaluation purposes)  
      
##### Usage Example
To run an experiment.
  
```
cortex GAN --d.source CIFAR10 --d.copy_to_local
```

# Customize Cortex2.0 
Cortex is meant to be an API-minimal library for runnning machine learning experiments that use gradient descent and backprop (though in principle should extend beyond this). As such, the actual models as implemented usually only require using one method from a `data_handler` object. Otherwise, as an illustration of how to get cortex working for your models, first look at:  
https://github.com/rdevon/cortex2.0/blob/master/arch/classifier.py  
  
Note that this module only imports from .modules, which has some standard Pytorch neural networks. Nowhere is there an import from the cortex core library.  
  
classifier.py requires the following to be incorporated into cortex:  
  
* It needs to be in the `models` directory (This needs to be changed to make it more useable)  
* It needs a `DEFAULTS` global dictionary. This dictionary specifies the kwargs that will be used in the main procedure (see below) and `build_models`, as well as any defaults for `train`, `optimizer`, and `data`:  
  * `data`: data args. Used in `lib/data/setup`  
 * `optimizer`: optimizer args. Used in `lib/train/setup`  
 * `model`: model args. Used in the `build_model` method in the current module.  
  * `procedures`: procedures args. Used in the main procedure method in the current module.  
  * `train`: train args. Used in `lib/train/main_loop`  
* It needs to include a `build_model`. It needs to take as input the data handler and all model args specified in the `DEFAULTS.model`. This is where the Pytorch neural network modules are specified. This needs to return a dictionary of networks (values can be tuples, e.g., `dict(nets=(net1, net2))`), along with a pointer to the main procedure method.  
* It needs to define (or import) a procedure method. This method needs to take as input a dictionary of networks (as returned by `build_model`), the data handler, and all procedure args specified in `DEFAULTS.procedures`. It must return a tuple:  
  * Losses: either dictionary (with the same keys as the networks) or a loss. If it's a dictionary, only the networks with the same key will have the loss applied.  
  * Results: A dictionary of values that will be plotted  
  * Samples: A dictionary of things that will be visualized  
  * <Ignore>  
* Finally, the model needs to be registered by adding it to the `arch_names` string in `models/__init__.py`  
  


