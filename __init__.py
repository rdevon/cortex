'''

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import ast
import copy
import logging
import os
from os import path
from shutil import copyfile, rmtree

import torch

from lib import config, models, exp
from lib.log_utils import set_file_logger, set_stream_logger
from lib.utils import convert_nested_dict_to_handler, make_argument_parser, _protected_args
from lib import data, optimizer, train
from lib.viz import init as viz_init


logger = logging.getLogger('cortex.init')

_args = dict(data=data._args, optimizer=optimizer._args, train=train._args)
_args_help = dict(data=data._args_help, optimizer=optimizer._args_help, train=train._args_help)


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


_known_args = dict((k, list(v.keys())) for k, v in _args.items())


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
            k_base = None
            for i, k_ in enumerate(k_split):
                if i < len(k_split) - 1:
                    if k_base in _known_args and k_ not in kw:
                        if k_ in _known_args[k_base]:
                            kw[k_] = {}
                    if k_ in kw:
                        kw = kw[k_]
                        k_base = k_
                    else:
                        raise ValueError('Unknown arg {}'.format(k))
                else:
                    if kw is None:
                        kw = dict(k_=v)
                    else:
                        kw[k_] = v


def handle_deprecated(data=None, model=None, optimizer=None, routines=None, test_routines=None, train=None):
    if 'noise_variables' in data:
        for k, v in data['noise_variables'].items():
            if isinstance(v, tuple):
                DeprecationWarning('Old-stype tuple found in noise argument. Converting.')
                data['noise_variables'][k] = dict(dist=v[0], size=v[1])

    if 'updates_per_model' in optimizer:
        DeprecationWarning('Old-stype `updates_per_model` found. Use `updates_per_routine`')
        optimizer['updates_per_routine'] = optimizer.pop('updates_per_model')


def setup_device(device):
    if torch.cuda.is_available():
        if device < torch.cuda.device_count():
            logger.info('Using GPU ' + str(device))
            exp.DEVICE = torch.device(device)
        else:
            logger.info('GPU ' + str(device) + ' doesn\'t exists. Using CPU')
    else:
        logger.info('Using CPU')

def setup_reload(arch, exp_file):
        
    setup_device(d['args']['device'])

    config_file_path = path.join(path.dirname(
        path.abspath(__file__)), 'config.yaml')
    if not path.isfile(config_file_path): config_file_path = None
    config.update_config(config_file_path)
    viz_init()

    arch.setup(arch)

    logger.info('Reloading from {}'.format(exp_file))
    d = torch.load(exp_file)
    exp.INFO.update(**d['info'])
    exp.NAME = d['info']['name']
    exp.SUMMARY.update(**d['summary'])
    exp.ARGS.update(**d['args'])
    reloads = d['arch'].keys()
    for k in reloads:
        exp.MODEL_PARAMS_RELOAD.update(**{k: d['arch'][k]})


def reload_experiment(args):
    exp_file = args.reload
    reloads = args.reloads
    name = args.name

    d = torch.load(exp_file)
    exp.INFO.update(**d['info'])
    exp.NAME = d['info']['name']
    exp.SUMMARY.update(**d['summary'])
    kwargs = convert_nested_dict_to_handler(d['args'])
    handle_deprecated(**kwargs)
    exp.ARGS.update(**kwargs)
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


def setup():

    # Parse args and set logger, cuda
    parser = make_argument_parser()
    for arg_k in _args:
        args = _args[arg_k]
        for k, v in args.items():
            if isinstance(v, dict):
                pass
            type_ = type(v) if v is not None else str
            parser.add_argument('--' + arg_k + '.' + k, default=None, type=type_, help=_args_help[arg_k][k])
    args = parser.parse_args()

    set_stream_logger(args.verbosity)

    setup_device(args.device)

    # Setup file paths
    config_file_path = path.join(path.dirname(
        path.abspath(__file__)), 'config.yaml')
    if not path.isfile(config_file_path): config_file_path = None

    # User config
    config.update_config(config_file_path)

    # Initialize visualizer
    viz_init()

    # Set up the architecture, build the models
    arch = models.setup(args.arch)

    if args.reload:
        # Reload the old experiment
        if not path.isfile(args.reload):
            raise ValueError('Cannot find {}'.format(args.reload))
        logger.info('Reloading from {}'.format(args.reload))
        copyfile(args.reload, args.reload + '.bak')
        reload_experiment(args)
        for k, v in vars(args).items():
            if v is not None:
                if '.' in k:
                    head, tail = k.split('.')
                    exp.ARGS[head][tail] = v
        update_args(args.args, **exp.ARGS)

    else:
        # Make a new experiment
        kwargs = copy.deepcopy(_args)
        for k, v in arch.DEFAULT_CONFIG.items():
            if k not in kwargs:
                kwargs[k] = {}
            kwargs[k].update(**v)
        for k, v in vars(args).items():
            if v is not None:
                if '.' in k:
                    head, tail = k.split('.')
                    kwargs[head][tail] = v

        kwargs = convert_nested_dict_to_handler(kwargs)

        if 'test_routines' not in kwargs.keys():
            kwargs['test_routines'] = {}

        update_args(args.args, **kwargs)

        name = args.name
        if name is None:
            name = args.arch
        exp.NAME = name
        exp.INFO['name'] = name
        exp.configure_experiment(config_file=args.config_file, **kwargs)
        exp.ARGS.update(**kwargs)
        setup_out_dir(args.out_path, name, clean=args.clean)

    if hasattr(arch, 'SETUP'):
        getattr(arch, 'SETUP')(**exp.ARGS)

    if hasattr(arch, 'DataLoader'):
        DataLoader = getattr(arch, 'DataLoader')
        logger.info('Loading custom DataLoader class, {}'.format(DataLoader))
        exp.ARGS['data']['DataLoader'] = DataLoader

    if hasattr(arch, 'Dataset'):
        Dataset = getattr('Dataset')
        logger.info('Loading custom Dataset class, {}'.format(Dataset))
        exp.ARGS['data']['Dataset'] = Dataset

    if hasattr(arch, 'transform'):
        transform = getattr('transform')
        logger.info('Loading custom transform function, {}'.format(transform))
        exp.ARGS['data']['transform'] = transform

    exp.ARGS['data']['copy_to_local'] = args.copy_to_local

    exp.ARGS['train']['test_mode'] = args.test
