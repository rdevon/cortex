'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import logging
from os import path
import yaml

import torch


logger = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
USE_CUDA = False
SUMMARY = {'train': {}, 'test': {}}
OUT_DIRS = {}
ARGS = {}
INFO = {'name': NAME, 'epoch': 0}
MODEL_PARAMS_RELOAD = {}

# Models criteria and results
MODELS = {}
CRITERIA = {}
PROCEDURES = {}


def file_string(prefix=''):
    if prefix == '': return NAME
    return '{}({})'.format(NAME, prefix)


def configure_experiment(data=None, model=None, optimizer=None, train=None, procedures=None, test_procedures=None,
                         config_file=None):
    '''Loads arguments into a yaml file.

    '''
    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        if model is not None: model.update(**d.get('model', {}))
        if optimizer is not None: optimizer.update(**d.get('optimizer', {}))
        if train is not None: train.update(**d.get('train', {}))
        if data is not None: data.update(**d.get('data', {}))
        if procedures is not None: procedures.update(**d.get('procedures', {}))
        if test_procedures is not None: test_procedures.update(**d.get('test_procedures', procedures))

    for k, v in procedures.items():
        if not k in test_procedures.keys():
            test_procedures[k] = v

    logger.info('Training model with: \n\tdata args {}, \n\toptimizer args {} '
                '\n\tmodel args {} \n\ttrain args {} \n\tprocedure args {} \n\ttest procedure args {}'.format(
        data, optimizer, model, train, procedures, test_procedures))


def save(prefix=''):
    prefix = file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    models = {}
    for k, model in MODELS.items():
        if isinstance(model, (tuple, list)):
            nets = []
            for net in model:
                nets.append(net)#(net.module if USE_CUDA else net)
            models[k] = nets
        else:
            models[k] = model#.module if USE_CUDA else model

    state = dict(
        models=models,
        info=INFO,
        args=ARGS,
        out_dirs=OUT_DIRS,
        summary=SUMMARY
    )

    file_path = path.join(binary_dir, '{}.t7'.format(prefix))
    logger.info('Saving checkpoint {}'.format(file_path))
    torch.save(state, file_path)


def setup(models, procedures):
    global MODELS, PROCEDURES
    MODELS.update(**models)
    if not isinstance(procedures, dict):
        procedures = dict(main=procedures)
    PROCEDURES.update(**procedures)

    if MODEL_PARAMS_RELOAD:
        reload_models()


def reload_models():
    global MODELS
    for k in MODELS.keys():
        v_ = MODEL_PARAMS_RELOAD.get(k, None)
        if v_:
            logger.info('Reloading model {}'.format(k))
            logger.debug(v_)
            MODELS[k] = v_