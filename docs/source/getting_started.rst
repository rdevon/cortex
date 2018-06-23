Getting Started
===============

Configuration
~~~~~~~~~~~~~

Visdom Server
^^^^^^^^^^^^^

Start a Visdom server and look for server address in the output. By
default, the server's address is ``http://localhost:8097``.

::

    $python -m visdom.server

Experiment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

The first thing to do is to set up the config.yaml. This file is
user-specific (it got tracked at some point, so I need to fix this), and
will tell cortex everything user-specific regarding data locations,
visualation, and outputs.

::

    $rm -rf ~/.cortex.yml
    $cortex setup

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
   | ### Usage ##### Help

   cortex --help ##### Options

There are many command-line options in cortex:

-  ``-n`` Name of experiment. Experiment is saved in /
-  ``-S`` data Source (from torchvision or user-specified in
   config.yaml)
-  ``-r`` Reload model (takes the .t7 file as the argument)
-  ``-a`` Arguments. This is a ``,``-delimited string. For instance, to
   increase the training epochs on the above example and use sgd, this
   should work:
   ``-a train.epochs=5000,optimizer.optimizer=sgd``
-  ``-c`` Config (For yamls of arguments, doesn't appear to be working
   right now)
-  ``-k`` Klean an experiment directory
-  ``-v`` Verbosity of logger
-  ``-o`` Out path (overrides ``config.yaml``)
-  ``-t`` Test mode (for evaluation purposes)

Usage Example
'''''''''''''

To run an experiment.

::

    cortex GAN --d.source CIFAR10 --d.copy_to_local
