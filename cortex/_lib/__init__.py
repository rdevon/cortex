'''Cortex setup

'''

import copy
import glob
import logging
import os
import pprint

from . import config, exp, log_utils, models
from .parsing import default_args, parse_args, update_args
from .viz import init as viz_init

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.init')


def setup_cortex(model=None):
    '''Sets up cortex

    Finds all the models in cortex, parses the command line, and sets the
    logger.

    Returns:
        TODO

    '''
    args = parse_args(models.MODEL_PLUGINS, model=model)

    log_utils.set_stream_logger(args.verbosity)

    return args


def find_autoreload(out_path, global_out_path, name):
    out_path = out_path or global_out_path
    out_path = os.path.join(out_path, name)
    binary_dir = os.path.join(out_path, 'binaries')
    binaries = glob.glob(os.path.join(binary_dir, '*.t7'))
    binaries.sort(key=os.path.getmtime)

    if len(binaries) > 0:
        return binaries[-1]
    else:
        logger.warning('No model found to auto-reload')
        return None


def setup_experiment(args, model=None, testmode=False):
    '''Sets up the experiment

    Args:
        args: TODO

    '''

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

    experiment_args = copy.deepcopy(default_args)
    update_args(experiment_args, exp.ARGS)

    if not testmode:
        viz_init(config.CONFIG.viz)

    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
            elif k in model.kwargs:
                head = 'model'
                tail = k
            else:
                continue
            exp.ARGS[head][tail] = v

    reload_nets = None

    if args.autoreload:
        reload_path = find_autoreload(args.out_path, config.CONFIG.out_path,
                                      args.name or model_name)
    elif args.reload:
        reload_path = args.reload
    else:
        reload_path = None

    if reload_path:
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
    else:
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

    update_nested_dicts(exp.ARGS['model'], model.kwargs)
    exp.ARGS['model'].update(**model.kwargs)

    exp.configure_from_yaml(config_file=args.config_file)

    for k, v in exp.ARGS.items():
        logger.info('Ultimate {} arguments: \n{}'
                    .format(k, pprint.pformat(v)))

    return model, reload_nets
