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
MODEL_HANDLER = None
CRITERIA = {}
TRAIN_ROUTINES = {}
TEST_ROUTINES = {}
FINISH_TRAIN_ROUTINES = {}
FINISH_TEST_ROUTINES = {}


def file_string(prefix=''):
    if prefix == '': return NAME
    return '{}({})'.format(NAME, prefix)


def configure_experiment(data=None, model=None, optimizer=None, train=None, routines=None, test_routines=None,
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
        if routines is not None: routines.update(**d.get('routine', {}))
        if test_routines is not None: test_routines.update(**d.get('test_routines', {}))

    for k, v in routines.items():
        if not k in test_routines.keys():
            test_routines[k] = v

    logger.info('Training model with: \n\tdata args {}, \n\toptimizer args {} '
                '\n\tmodel args {} \n\ttrain args {} \n\troutine args {} \n\ttest routine args {}'.format(
        data, optimizer, model, train, routines, test_routines))


def save(prefix=''):
    prefix = file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    models = {}
    for k, model in MODEL_HANDLER.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            nets = []
            for net in model:
                nets.append(net)
            models[k] = nets
        else:
            models[k] = model

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


def setup(models, train_routines=None, test_routines=None, finish_train_routines=None, finish_test_routines=None):
    global MODEL_HANDLER, TRAIN_ROUTINES, TEST_ROUTINES, FINISH_TRAIN_ROUTINES, FINISH_TEST_ROUTINES

    MODEL_HANDLER = models
    TRAIN_ROUTINES.update(**train_routines)
    TEST_ROUTINES.update(**test_routines)
    FINISH_TRAIN_ROUTINES.update(**finish_train_routines)
    FINISH_TEST_ROUTINES.update(**finish_test_routines)

    if MODEL_PARAMS_RELOAD:
        reload_models()


def reload_models():
    global MODEL_HANDLER
    if MODEL_HANDLER is None:
        raise RuntimeError('MODEL_HANDLER not set. `reload_models` should only be used after '
                           '`models.setup_models` has been called.')
    for k, v in MODEL_PARAMS_RELOAD.items():
        logger.info('Reloading model {}'.format(k))
        logger.debug(v)
        MODEL_HANDLER[k] = v