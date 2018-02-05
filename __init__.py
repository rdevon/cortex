'''

'''

import ast
import logging
import os
from os import path
from shutil import copyfile, rmtree

import numpy as np
import torch

import models
from lib import config, exp
from lib.log_utils import set_file_logger, set_stream_logger
from lib.utils import make_argument_parser
from lib.viz import init as viz_init


logger = logging.getLogger('cortex.init')


def setup_out_dir(out_path, name=None, clean=False):
    '''Sets up the output directory of an experiment.

    '''

    if out_path is None:
        if name is None:
            raise ValueError('If `out_path` (-o) argument is not set, you '
                             'must set the `name` (-n)')
        out_path = config.OUT_PATH
        if out_path is None:
            raise ValueError('If `--out_path` (`-o`) argument is not set, you '
                             'must set both the name argument and configure '
                             'the out_path entry in `config.yaml`')

    if name is not None: out_path = path.join(out_path, name)

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

    if not path.isdir(binary_dir): os.mkdir(binary_dir)
    if not path.isdir(image_dir): os.mkdir(image_dir)

    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))

    exp.OUT_DIRS.update(binary_dir=binary_dir, image_dir=image_dir)


def setup_reload(exp_file):
    config_file_path = path.join(path.dirname(path.dirname(
        path.abspath(__file__))), 'config.yaml')
    if not path.isfile(config_file_path): config_file_path = None
    config.update_config(config_file_path)
    viz_init()
    reload_experiment(exp_file)


def reload_experiment(args):
    exp_file = args.reload
    reloads = args.reloads
    name = args.name

    d = torch.load(exp_file)
    exp.INFO.update(**d['info'])
    exp.NAME = d['info']['name']
    exp.SUMMARY.update(**d['summary'])
    exp.ARGS.update(**d['args'])
    reloads = reloads or d['models'].keys()
    for k in reloads:
        exp.MODEL_PARAMS_RELOAD.update(**{k: d['models'][k]})
    out_dirs = d['out_dirs']

    if name:
        exp.NAME = name
        exp.INFO['name'] = name
        setup_out_dir(args.out_path, name, clean=args.clean)
    out_path = path.dirname(path.dirname(exp_file))
    out_dirs = dict((k, path.join(out_path, path.basename(v))) for k, v in out_dirs.items())
    exp.OUT_DIRS.update(**out_dirs)


def update_args(args, **kwargs):
    if args is not None:
        a = args.split(',')
        for a_ in a:
            k, v = a_.split('=')

            try:
                v = ast.literal_eval(v)
            except ValueError:
                pass

            k_split = k.split('.')
            kw = kwargs
            for i, k_ in enumerate(k_split):
                if i < len(k_split) - 1:
                    if k_ in kw:
                        kw = kw[k_]
                    else:
                        raise ValueError('Unknown arg {}'.format(k))
                else:
                    kw[k_] = v


def setup(use_cuda):
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)
    exp.USE_CUDA = use_cuda

    if exp.USE_CUDA:
        logger.info('Using GPU')
    else:
        logger.info('Using CPU')

    config_file_path = path.join(path.dirname(
        path.abspath(__file__)), 'config.yaml')
    if not path.isfile(config_file_path): config_file_path = None
    config.update_config(config_file_path)
    viz_init()
    arch = models.setup(args.arch)

    if args.reload:
        if not path.isfile(args.reload):
            raise ValueError('Cannot find {}'.format(args.reload))
        logger.info('Reloading from {}'.format(args.reload))
        copyfile(args.reload, args.reload + '.bak')
        reload_experiment(args)
        update_args(args.args, **exp.ARGS)

    else:
        kwargs = {}
        for k, v in arch.DEFAULTS.items():
            kwargs[k] = {}
            kwargs[k].update(**v)

        if 'test_procedures' not in kwargs.keys():
            kwargs['test_procedures'] = {}

        kwargs['data']['source'] = args.source
        update_args(args.args, **kwargs)

        name = args.name
        if name is None:
            name = args.arch
        exp.NAME = name
        exp.INFO['name'] = name
        exp.configure_experiment(config_file=args.config_file, **kwargs)
        exp.ARGS.update(**kwargs)
        setup_out_dir(args.out_path, name, clean=args.clean)

    if hasattr(arch, 'setup'):
        getattr(arch, 'setup')(**exp.ARGS)

    exp.ARGS['train']['test_mode'] = args.test
