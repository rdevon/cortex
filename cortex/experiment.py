"""
Experiment module.

Used for saving, loading, summarizing, etc
"""

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
import os
from os import path
from shutil import copyfile, rmtree
import yaml
import torch
from . import models
from cortex.utils.log_utils import set_file_logger
from cortex import Handler, convert_nested_dict_to_handler

LOGGER = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
SUMMARY = {'train': {}, 'test': {}}
OUT_DIRS = {}
ARGS = Handler(data=Handler(), model=Handler(), optimizer=Handler(), routines=Handler(), test_routines=Handler(),
               train=Handler())
INFO = {'name': NAME, 'epoch': 0}
DEVICE = torch.device('cpu')

# Models criteria and results
CRITERIA = {}
TRAIN_ROUTINES = {}
TEST_ROUTINES = {}
FINISH_TRAIN_ROUTINES = {}
FINISH_TEST_ROUTINES = {}


def _file_string(prefix=''):
    if prefix == '': return NAME
    return '{}({})'.format(NAME, prefix)


def _handle_deprecated(data=None, model=None, optimizer=None, routines=None, test_routines=None, train=None):
    """
    TODO
    :param data:
    :type data:
    :param model:
    :type model:
    :param optimizer:
    :type optimizer:
    :param routines:
    :type routines:
    :param test_routines:
    :type test_routines:
    :param train:
    :type train:
    """
    if 'noise_variables' in data:
        for k, v in data['noise_variables'].items():
            if isinstance(v, tuple):
                LOGGER.warning('Old-stype tuple found in noise argument. Converting.')
                data['noise_variables'][k] = dict(dist=v[0], size=v[1])

    if 'updates_per_model' in optimizer:
        LOGGER.warning('Old-stype `updates_per_model` found. Use `updates_per_routine`')
        optimizer['updates_per_routine'] = optimizer.pop('updates_per_model')

    if 'setup_fn' in data:
        LOGGER.warning('Unused `setup_fn` found in `data` args. Ignoring')
        data.pop('setup_fn')


def update_args(kwargs):
    """
    TODO
    :param kwargs:
    :type kwargs:
    """
    global ARGS
    def _update_args(from_kwargs, to_kwargs):
        for k, v in from_kwargs.items():
            if k not in to_kwargs:
                to_kwargs[k] = v
            else:
                if isinstance(v, dict) and isinstance(to_kwargs[k], dict):
                    _update_args(v, to_kwargs[k])
                else:
                    to_kwargs[k] = v

    kwargs = convert_nested_dict_to_handler(kwargs)
    for k in kwargs:
        if k not in ARGS:
            raise KeyError('Argument key {} not supported. Available: {}'.format(k, tuple(ARGS.keys())))
        elif not isinstance(kwargs[k], dict):
            raise ValueError('Only dictionaries supported for base values.')
        else:
            _update_args(kwargs[k], ARGS[k])


def copy_test_routines():
    """
    TODO
    """
    global ARGS
    for k, v in ARGS.routines.items():
        if not k in ARGS.test_routines.keys():
            ARGS.test_routines[k] = v


def configure_from_yaml(config_file=None):
    """
    Loads arguments into a yaml file.
    :param config_file:
    :type config_file:
    """
    global ARGS
    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        LOGGER.info('Loading config {}'.format(d))
        ARGS.model.update(**d.get('model', {}))
        ARGS.optimizer.update(**d.get('optimizer', {}))
        ARGS.train.update(**d.get('train', {}))
        ARGS.data.update(**d.get('data', {}))
        ARGS.routines.update(**d.get('routine', {}))
        ARGS.test_routines.update(**d.get('test_routines', {}))


def setup_new(arch_default_args, name, out_path, clean, config, model_file, reloads):
    """
    TODO
    :param arch_default_args:
    :type arch_default_args:
    :param name:
    :type name:
    :param out_path:
    :type out_path:
    :param clean:
    :type clean:
    :param config:
    :type config:
    :param model_file:
    :type model_file:
    :param reloads:
    :type reloads:
    """
    global NAME, INFO
    update_args(arch_default_args)

    NAME = name
    INFO['name'] = name
    setup_out_dir(out_path, config.out_path, name, clean=clean)

    if model_file:
        d = torch.load(model_file)
        reloads = reloads or d['models'].keys()
        for k in reloads:
            models.reload_models(**{k: d['models'][k]})
            if isinstance(d['models'][k], list):
                for m in d['models'][k]:
                    m = m.to(DEVICE)
            else:
                d['models'][k] = d['models'][k].to(DEVICE)
            

def reload(exp_file, reloads, name, out_path, clean, config):
    """
    TODO
    :param exp_file:
    :type exp_file:
    :param reloads:
    :type reloads:
    :param name:
    :type name:
    :param out_path:
    :type out_path:
    :param clean:
    :type clean:
    :param config:
    :type config:
    """
    global ARGS, INFO, MODEL_PARAMS_RELOAD, NAME, OUT_DIRS, SUMMARY

    if not path.isfile(exp_file):
        raise ValueError('Cannot find {}'.format(exp_file))

    LOGGER.info('Reloading from {} and creating backup'.format(exp_file))
    copyfile(exp_file, exp_file + '.bak')

    d = torch.load(exp_file)
    info = d['info']
    if not name:
        name = info['name']
    summary = d['summary']
    args = d['args']
    out_dirs = d['out_dirs']

    INFO.update(**info)
    NAME = name
    SUMMARY.update(**summary)

    _handle_deprecated(**args)
    update_args(args)

    reloads = reloads or d['models'].keys()
    for k in reloads:
        models.reload_models(**{k: d['models'][k]})
        if isinstance(d['models'][k], list):
            for m in d['models'][k]:
                m = m.to(DEVICE)
        else:
            d['models'][k] = d['models'][k].to(DEVICE)

    if name:
        NAME = name
        INFO['name'] = name
        setup_out_dir(out_path, config.out_path, name, clean=clean)

    out_path = path.dirname(path.dirname(exp_file))
    out_dirs = dict((k, path.join(out_path, path.basename(v))) for k, v in out_dirs.items())
    OUT_DIRS.update(**out_dirs)


def save(prefix=''):
    """
    TODO
    :param prefix:
    :type prefix:
    :return:
    :rtype:
    """
    prefix = _file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    models_= {}
    for k, model in models.MODEL_HANDLER.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            nets = []
            for net in model:
                nets.append(net)
            models_[k] = nets
        else:
            models_[k] = model

    state = dict(
        models=models_,
        info=INFO,
        args=ARGS,
        out_dirs=OUT_DIRS,
        summary=SUMMARY
    )

    file_path = path.join(binary_dir, '{}.t7'.format(prefix))
    LOGGER.info('Saving checkpoint {}'.format(file_path))
    torch.save(state, file_path)


def setup_out_dir(out_path, global_out_path, name=None, clean=False):
    """
    Sets up the output directory of an experiment.
    :param out_path:
    :type out_path:
    :param global_out_path:
    :type global_out_path:
    :param name:
    :type name:
    :param clean:
    :type clean:
    """
    global OUT_DIRS

    if out_path is None:
        if name is None:
            raise ValueError('If `out_path` (-o) argument is not set, you '
                             'must set the `name` (-n)')
        out_path = global_out_path
        if out_path is None:
            raise ValueError('If `--out_path` (`-o`) argument is not set, you '
                             'must set both the name argument and configure '
                             'the out_path entry in `config.yaml`')

    if name is not None: out_path = path.join(out_path, name)

    if not path.isdir(out_path):
        LOGGER.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)

    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')

    if clean:
        LOGGER.warning('Cleaning directory (cannot be undone)')
        if path.isdir(binary_dir):
            rmtree(binary_dir)
        if path.isdir(image_dir):
            rmtree(image_dir)

    if not path.isdir(binary_dir): os.mkdir(binary_dir)
    if not path.isdir(image_dir): os.mkdir(image_dir)

    LOGGER.info('Setting out path to `{}`'.format(out_path))
    LOGGER.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))

    OUT_DIRS.update(binary_dir=binary_dir, image_dir=image_dir)


def setup_device(device):
    """
    :param device:
    :type device:
    """
    global DEVICE
    if torch.cuda.is_available():
        if device < torch.cuda.device_count():
            LOGGER.info('Using GPU ' + str(device))
            DEVICE = torch.device(device)
        else:
            LOGGER.info('GPU ' + str(device) + ' doesn\'t exists. Using CPU')
    else:
        LOGGER.info('Using CPU')


def setup_routines(train_routines=None, test_routines=None, finish_train_routines=None, finish_test_routines=None):
    """
    :param train_routines:
    :type train_routines:
    :param test_routines:
    :type test_routines:
    :param finish_train_routines:
    :type finish_train_routines:
    :param finish_test_routines:
    :type finish_test_routines:
    """
    global TRAIN_ROUTINES, TEST_ROUTINES, FINISH_TRAIN_ROUTINES, FINISH_TEST_ROUTINES

    TRAIN_ROUTINES.update(**train_routines)
    TEST_ROUTINES.update(**test_routines)
    FINISH_TRAIN_ROUTINES.update(**finish_train_routines)
    FINISH_TEST_ROUTINES.update(**finish_test_routines)