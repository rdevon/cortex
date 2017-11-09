'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import logging
from os import path

import numpy as np

logger = logging.getLogger('BGAN.exp')

# Experiment info
NAME = 'X'
RESULTS_EPOCHS = {}
RESULTS_UPDATES = {}
OUT_DIRS = {}
ARGS = {}
INFO = {'name': NAME, 'epoch': 0}
MODEL_PARAMS_RELOAD = None
ITERATORS = {}

# Models and tensors
MODELS = {}
INPUTS = {}
LOSSES = {}
STATS = {}
SAMPLES = {}
HISTOGRAMS = {}
TENSORS = {}


def file_string(prefix=''):
    if prefix == '': return NAME
    return '{}({})'.format(NAME, prefix)


def configure_experiment(data=None, model=None, loss=None, optimizer=None,
                         train=None, config_file=None):
    '''Loads arguments into a yaml file.

    '''
    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        if model is not None: model.update(**d.get('model', {}))
        if loss is not None: loss.update(**d.get('loss', {}))
        if optimizer is not None: optimizer.update(**d.get('optimizer', {}))
        if train is not None: train.update(**d.get('train', {}))
        if data is not None: data.update(**d.get('data', {}))

    logger.info('Training model with: \n\tdata args {}, \n\toptimizer args {} '
                '\n\tmodel args {} \n\tloss args {} \n\ttrain args {}'.format(
        data, optimizer, model, loss, train))


def load_tensors(inputs, losses, stats, samples, histograms, tensors):
    global INPUTS, LOSSES, STATS, SAMPLES, HISTOGRAMS, TENSORS

    for k, v in losses.items():
        stats[k][k + '_loss'] = v

    INPUTS.update(**inputs)
    LOSSES.update(**losses)
    STATS.update(**stats)
    SAMPLES.update(**samples)
    HISTOGRAMS.update(**histograms)
    TENSORS.update(**tensors)

    all_inputs = set([i for k, inp in inputs.items() for i in inp])
    INPUTS['all'] = all_inputs


def save(prefix=''):
    prefix = file_string(prefix)

    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return
    models = dict((k, lasagne.layers.get_all_param_values(v)) for k, v in MODELS.items())
    np.savez(path.join(binary_dir, '{}.npz'.format(prefix)),
             results_epochs=RESULTS_EPOCHS, results_updates=RESULTS_UPDATES,
             info=INFO, args=ARGS, models=models, out_dirs=OUT_DIRS)


def set_models(models):
    global MODELS
    MODELS.update(**models)
    if MODEL_PARAMS_RELOAD is not None:
        reload_params()


def reload_params():
    for k, v in MODELS.items():
        params = MODEL_PARAMS_RELOAD.get(k, None)
        if params:
            logger.info('Reloading parameters for {}'.format(k))
            lasagne.layers.set_all_param_values(v, params)
