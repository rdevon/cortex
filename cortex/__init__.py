'''

'''

import ast
import logging
import os
from os import path
from shutil import copyfile, rmtree

import numpy as np

from . import config, exp
from .log_utils import set_file_logger, set_stream_logger
from .utils import make_argument_parser
from .viz import init as viz_init


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


def reload_experiment(exp_file):
    d = np.load(exp_file)
    exp.INFO.update(**d['info'][()])
    exp.NAME = d['info'][()]['name']
    exp.RESULTS_EPOCHS.update(**d['results_epochs'][()])
    exp.RESULTS_UPDATES.update(**d['results_updates'][()])
    exp.ARGS.update(**d['args'][()])
    exp.MODEL_PARAMS_RELOAD = d['models'][()]
    out_dirs = d['out_dirs'][()]
    out_path = path.dirname(path.dirname(exp_file))
    out_dirs = dict((k, path.join(out_path, path.basename(v))) for k, v in out_dirs.items())
    exp.OUT_DIRS.update(**out_dirs)


def setup():
    parser = make_argument_parser()
    args = parser.parse_args()
    set_stream_logger(args.verbosity)

    config_file_path = path.join(path.dirname(path.dirname(
        path.abspath(__file__))), 'config.yaml')
    if not path.isfile(config_file_path): config_file_path = None
    config.update_config(config_file_path)
    viz_init()
    assert False

    if args.reload:
        if not path.isfile(args.reload):
            raise ValueError('Cannot find {}'.format(args.reload))
        logger.info('Reloading from {}'.format(args.reload))
        copyfile(args.reload, args.reload + '.bak')
        reload_experiment(args.reload)
    else:
        kwargs = {}
        for k, v in _default_args.items():
            kwargs[k] = {}
            kwargs[k].update(**v)

        kwargs['data']['source'] = args.source
        kwargs['data']['meta'] = args.meta
        if args.args is not None:
            a = args.args.split(',')
            for a_ in a:
                k, v = a_.split('=')
                k, k_ = k.split('.')
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    pass
                if k in kwargs:
                    kwargs[k][k_] = v
                else:
                    raise ValueError('Unknown arg {}'.format(k))

        kwargs['data']['use_tanh'] = bool(kwargs['model']['nonlin'] == 'tanh')
        name = args.name

        if name is None:
            name = out_paths['binary_dir'].split('/')[-2]
        exp.NAME = name
        exp.INFO['name'] = name
        exp.configure_experiment(config_file=args.config_file, **kwargs)
        exp.ARGS.update(**kwargs)
        setup_out_dir(args.out_path, name, clean=args.clean)

