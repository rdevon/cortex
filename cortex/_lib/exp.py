'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import logging
import os
from os import path
from shutil import copyfile, rmtree
import yaml

import torch

from . import models
from .log_utils import set_file_logger
from .handlers import Handler, convert_nested_dict_to_handler

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
SUMMARY = {'train': {}, 'test': {}}
OUT_DIRS = {}
ARGS = Handler(
    data=Handler(),
    model=Handler(),
    optimizer=Handler(),
    train=Handler())
INFO = {'name': NAME, 'epoch': 0}
DEVICE = torch.device('cpu')


def _file_string(prefix=''):
    if prefix == '':
        return NAME
    return '{}({})'.format(NAME, prefix)


def update_args(kwargs):
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
            raise KeyError(
                'Argument key {} not supported. Available: {}'.format(
                    k, tuple(
                        ARGS.keys())))
        elif not isinstance(kwargs[k], dict):
            raise ValueError('Only dictionaries supported for base values.')
        else:
            _update_args(kwargs[k], ARGS[k])


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


def setup_new(
        arch_default_args,
        name,
        out_path,
        clean,
        config,
        model_file,
        reloads):
    global NAME, INFO
    update_args(arch_default_args)

    NAME = name
    INFO['name'] = name
    setup_out_dir(out_path, config.out_path, name, clean=clean)

    if model_file:
        d = torch.load(model_file)
        reloads = reloads or d['builds'].keys()
        for k in reloads:
            models.reload_models(**{k: d['builds'][k]})
            if isinstance(d['builds'][k], list):
                for m in d['builds'][k]:
                    m.to(DEVICE)
            else:
                d['builds'][k] = d['builds'][k].to(DEVICE)


def reload(exp_file, reloads, name, out_path, clean, config):
    global ARGS, INFO, MODEL_PARAMS_RELOAD, NAME, OUT_DIRS, SUMMARY

    if not path.isfile(exp_file):
        raise ValueError('Cannot find {}'.format(exp_file))

    logger.info('Reloading from {} and creating backup'.format(exp_file))
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

    update_args(args)

    reloads = reloads or d['builds'].keys()
    for k in reloads:
        models.reload_models(**{k: d['builds'][k]})
        if isinstance(d['builds'][k], list):
            for m in d['builds'][k]:
                m.to(DEVICE)
        else:
            d['builds'][k] = d['builds'][k].to(DEVICE)

    if name:
        NAME = name
        INFO['name'] = name
        setup_out_dir(out_path, config.out_path, name, clean=clean)

    out_path = path.dirname(path.dirname(exp_file))
    out_dirs = dict((k, path.join(out_path, path.basename(v)))
                    for k, v in out_dirs.items())
    OUT_DIRS.update(**out_dirs)


def save(prefix=''):
    prefix = _file_string(prefix)
    binary_dir = OUT_DIRS.get('binary_dir', None)
    if binary_dir is None:
        return

    models_ = {}
    for k, model in models.MODEL.nets.items():
        if k == 'extras':
            continue
        if isinstance(model, (tuple, list)):
            nets = []
            for net in model:
                nets.append(net)
            models_[k] = nets
        else:
            models_[k] = model

    def strip_Nones(d):
        d_ = {}
        for k, v in d.items():
            if isinstance(v, dict):
                d_[k] = strip_Nones(v)
            elif v is not None:
                d_[k] = v
        return d_

    args = strip_Nones(ARGS)

    state = dict(
        models=models_,
        info=INFO,
        args=args,
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
    if torch.cuda.is_available():
        if device < torch.cuda.device_count():
            logger.info('Using GPU ' + str(device))
            DEVICE = torch.device(device)
        else:
            logger.info('GPU ' + str(device) + ' doesn\'t exists. Using CPU')
    else:
        logger.info('Using CPU')
