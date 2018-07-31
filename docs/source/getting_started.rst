Getting Started
===============

Configuration
~~~~~~~~~~~~~

The first thing to do is to set up the config.yaml. This file is
user-specific (it got tracked at some point, so I need to fix this), and
will tell cortex everything user-specific regarding data locations,
visualation, and outputs.

::

    $ rm -rf ~/.cortex.yml
    $ cortex setup

Configuration File Example
''''''''''''''''''''''''''

Located at ``~/.cortex.yml``

.. code:: python

    torchvision_data_path: /data/milatmp1/hjelmdev/data/
    data_paths: {
     Imagenet-12: /data/lisa/data/ImageNet2012_jpeg, CelebA: /tmp/hjelmdev/CelebA}viz: {
     font: /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf, server: 'http://132.204.26.180'}
    out_path: /data/milatmp1/hjelmdev/outs/

These are as follows:

-  torchvision\_data\_path: the path to all torchvision-specific
   datasets (details can be found in torchvision.datasets)
-  data\_paths: user-specified custom datasets. Currently, only support
   is for image folders (a la imagenet), but other dataset types (e.g.,
   text) are planned in the near-future.
-  vis: visdom specific arguments.
-  | out\_path: Out path for experiment outputs


Usage
'''''

   cortex --help

Built-ins
'''''''''

:setup:
    Setup cortex configuration.

:GAN: Generative adversarial network.
:VAE: Variational autoencoder.
:AdversarialAutoencoder: Adversarial Autoencoder.
:ALI:  Adversarially learned inference.
:ImageClassification: Basic image classifier.
:GAN_MINE: GAN + MINE.

Options
'''''''
-h, --help         show this help message and exit
-o OUT_PATH, --out_path OUT_PATH          Output path directory. All model results will go
                                                      here. If a new directory, a new one will be
                                                      created, as long as parent exists.
-n NAME, --name NAME       Name of the experiment. If given, base name of
                                                      output directory will be `--name`. If not given,
                                                      name will be the base name of the `--out_path`
-r RELOAD, --reload RELOAD     Path to model to reload.

-M LOAD_MODELS, --load_models LOAD_MODELS          Path to model to reload. Does not load args, info,
                                                      etc

-m META, --meta META                                 TODO

-c CONFIG_FILE, --config_file CONFIG_FILE            Configuration yaml file. See `exps/` for examples

-k, --clean                                           Cleans the output directory. This cannot be undone!

-v VERBOSITY, --verbosity VERBOSITY                 Verbosity of the logging. (0, 1, 2)

-d DEVICE, --device DEVICE                           TODO


Usage Example
'''''''''''''

To run an experiment.

::

    cortex GAN --d.source CIFAR10 --d.copy_to_local

Custom models
'''''''''''''

It is possible to run experiments with custom models made with Pytorch under the Cortex framework. For doing so, the model has to
be added to the demos folder under the root of the project. You can have a look to the given demo autoencoder and classifier already
implemented. The main difference is that, rather than registering the plugins, the run function of main.py has to be called. For example,

::

    if __name__ == '__main__':
    classifier = MyClassifier()
    run(model=classifier)

To run an experiment with a custom model.

::

    python my_model.py --d.source <Dataset> --d.copy_to_local
