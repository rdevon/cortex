'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import logging
import os
from os import path
from shutil import copyfile, rmtree
import yaml

import torch

from .log_utils import set_file_logger

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
SUMMARY = {'train': {}, 'test': {}}
OUT_DIRS = {}
ARGS = dict(data=dict(), model=dict(), optimizer=dict(), train=dict())
INFO = {'name': NAME, 'epoch': 0}
DEVICE = torch.device('cpu')


def _file_string(prefix=''):
    if prefix == '':
        return NAME
    return '{}_{}'.format(NAME, prefix)


def configure_from_yaml(config_file=None):
    '''Loads arguments into a yaml file.

    '''
    global ARGS

    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        ARGS.model.update(**d.get('builds', {}))
        ARGS.optimizer.update(**d.get('optimizer', {}))
        ARGS.train.update(**d.get('train', {}))
        ARGS.data.update(**d.get('data', {}))


def reload_model(model_to_reload):
    if not path.isfile(model_to_reload):
        raise ValueError('Cannot find {}'.format(model_to_reload))

    logger.info('Reloading from {} and creating backup'.format(model_to_reload))
    copyfile(model_to_reload, model_to_reload + '.bak')

    return torch.load(model_to_reload, map_location='cpu')


def save(model, prefix=''):
    '''Saves a model.

    Args:
        model: Model to save.
        prefix: Prefix for the save file.

    '''
    prefix = _file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    def strip_Nones(d):
        d_ = {}
        for k, v in d.items():
            if isinstance(v, dict):
                d_[k] = strip_Nones(v)
            elif v is not None:
                d_[k] = v
        return d_

    for net in model.nets.values():
        if hasattr(net, 'states'):
            net.states.clear()

    state = dict(
        nets=dict(model.nets),
        info=INFO,
        args=ARGS,
        out_dirs=OUT_DIRS,
        summary=SUMMARY
    )

    file_path = path.join(binary_dir, '{}.t7'.format(prefix))
    logger.info('Saving checkpoint {}'.format(file_path))
    torch.save(state, file_path)


def setup_out_dir(out_path, global_out_path, name=None, clean=False):
    '''Sets up the output directory of an experiment.

    '''
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

    if name is not None:
        out_path = path.join(out_path, name)

    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)

    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')

    if clean:
        logger.warning('Cleaning directory (cannot be undone)')
        if path.isdir(binary_dir):
            rmtree(binary_dir)
        if path.isdir(image_dir):
            rmtree(image_dir)

    if not path.isdir(binary_dir):
        os.mkdir(binary_dir)
    if not path.isdir(image_dir):
        os.mkdir(image_dir)

    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))

    OUT_DIRS.update(binary_dir=binary_dir, image_dir=image_dir)


def setup_device(device):
    global DEVICE
    if torch.cuda.is_available() and device != 'cpu':
        if device < torch.cuda.device_count():
            logger.info('Using GPU {}'.format(device))
            DEVICE = torch.device(device)
        else:
            logger.info('GPU {} doesn\'t exists. Using CPU'.format(device))
    else:
        logger.info('Using CPU')
