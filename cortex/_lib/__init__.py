'''Cortex setup

'''

import copy
import glob
import logging
import os

from . import config, exp, log_utils, models
from .parsing import DEFAULT_ARGS, parse_args, update_args
from .utils import print_hypers


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.init')


def setup_cortex(model=None):
    '''Sets up cortex

    Finds all the models in cortex, parses the command line, and sets the
    logger.

    Returns:
        argparse.arguments: Arguments parsed from command line.

    '''
    args = parse_args(models.MODEL_PLUGINS, model=model)

    log_utils.set_stream_logger(args.verbosity)

    return args


def find_autoreload(out_path, global_out_path, name, idx=0):
    """Finds the latest model saved for autoreload.

    Iterates through models in from newest to oldest until one loads without error.

    Args:
        out_path: Path to search for models.
        global_out_path: Alternative global path to search.
        name: Name of experiment to reload.
        idx: Index of model loaded.

    Returns:
        str: Path to binary.

    """
    out_path = out_path or global_out_path
    out_path = os.path.join(out_path, name)
    binary_dir = os.path.join(out_path, 'binaries')
    binaries = glob.glob(os.path.join(binary_dir, '*.t7'))
    binaries.sort(key=os.path.getmtime)

    if idx >= len(binaries):
        raise StopIteration

    return binaries[-(1 + idx)]


def setup_experiment(args, model=None, testmode=False):
    """Sets up the experiment

    Loads all the arguments from the arguments parsed from command line,
    Reloads models (if necessary), and propagates arguments to the main experiment
    dictionary.

    Args:
        argparse.arguments: Output from argument parser.

    """

    def update_nested_dicts(from_d, to_d):
        for k, v in from_d.items():
            if (k in to_d) and isinstance(to_d[k], dict):
                if not isinstance(v, dict):
                    raise ValueError('Updating dict entry with non-dict.')
                update_nested_dicts(v, to_d[k])
            else:
                to_d[k] = v

    exp.setup_device(args.device)

    if model is None:
        model_name = args.command
        model = models.get_model(model_name)
    else:
        model_name = model.__class__.__name__

    experiment_args = copy.deepcopy(DEFAULT_ARGS)
    update_args(experiment_args, exp.ARGS)

    exp.setup_visualization(args.visualization)
    
    if args.visualization == 'visdom':
        from .viz import init as viz_init
        if not testmode and args.visualization != 'off':
            viz_init(config.CONFIG.viz)
    

    

    def _expand_model_hypers(args, model):
        d = {}
        arg_keys = list(args.keys())

        for k in arg_keys:
            if k in model.kwargs.keys():
                v = args.pop(k)
                d[k] = v

        for child in model._models:
            child_args = dict()
            arg_keys = list(args.keys())
            for k in arg_keys:
                if k.split('.')[0] == child.name:
                    v = args.pop(k)
                    child_args['.'.join(k.split('.')[1:])] = v

            d_ = _expand_model_hypers(child_args, child)
            d[child.name] = d_

        return d

    def _pull_data_args(args):
        arg_keys = list(args.keys())
        d = {}
        for k in arg_keys:
            try:
                a = k.split('.')
                if len(a) == 3 and a[0] == 'data_args':
                    name, key = a[1], a[2]
                    value = args.pop(k)
                    if name not in d.keys():
                        d[name] = {}
                    d[name][key] = value

            except:
                pass

        return d

    model_hypers = _expand_model_hypers(vars(args), model)
    exp.ARGS['model'] = model_hypers

    data_args = _pull_data_args(vars(args))
    exp.ARGS['data'].update(**data_args)
    exp.ARGS['data'].update(((exp.ARGS['data'].pop('data_args', {}))))

    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
            else:
                continue

            exp.ARGS[head][tail] = v

    reload_nets = None

    def reload(reload_path):
        d = exp.reload_model(reload_path)
        exp.INFO.update(**d['info'])
        exp.NAME = exp.INFO['name']
        exp.SUMMARY.update(**d['summary'])
        update_nested_dicts(d['args'], exp.ARGS)

        if args.name:
            exp.INFO['name'] = exp.NAME
        if args.out_path or args.name:
            exp.setup_out_dir(args.out_path, config.CONFIG.out_path, exp.NAME,
                              clean=args.clean)
        else:
            exp.OUT_DIRS.update(**d['out_dirs'])

        reload_nets = d['nets']

        return reload_nets

    if args.autoreload:
        idx = 0
        while True:
            try:
                reload_path = find_autoreload(args.out_path, config.CONFIG.out_path,
                                              args.name or model_name, idx=idx)
                reload_nets = reload(reload_path)
                reload_path = True
                break
            except StopIteration:
                logger.warning('No suitable files found to autoreload. '
                               'Starting from scratch.')
                reload_path = False
                break
            except Exception as e:
                logger.warning(
                    'Loading error occurred ({}). Trying previous.'
                    .format(e))
                idx += 1

    elif args.reload:
        try:
            reload_nets = reload(args.reload)
            reload_path = True
        except:
            logger.warning('No suitable files found to autoreload. '
                           'Starting from scratch.')
            reload_path = False

    else:
        reload_path = False

    if not reload_path:
        if args.load_networks:
            d = exp.reload_model(args.load_networks)
            keys = args.networks_to_reload or d['nets']
            for key in keys:
                if key not in d['nets']:
                    raise KeyError('Model {} has no network called {}'
                                   .format(args.load_networks, key))
            reload_nets = dict((k, d['nets'][k]) for k in keys)

        exp.NAME = args.name or model_name
        exp.INFO['name'] = exp.NAME
        exp.setup_out_dir(args.out_path, config.CONFIG.out_path, exp.NAME,
                          clean=args.clean)


    if args.visualization == 'tensorboard':
        from .tensorborad import init as tb_init
        if not testmode:
            tb_init(exp.OUT_DIRS['tb'])

    str = print_hypers(exp.ARGS, s='Final hyperparameters: ', mode=args.visualization)
    logger.info(str)
    model.push_hyperparameters(exp.ARGS['model'])

    return model, reload_nets, args.lax_reload
