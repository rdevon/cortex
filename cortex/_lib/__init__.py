'''Cortex setup

'''

import copy
import logging
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


def setup_experiment(args, model=None, testmode=False):
    '''Sets up the experiment

    Args:
        args: TODO

    '''
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
    reload_nets = None
    if args.reload:
        d = exp.reload_model(args.reload)

        exp.INFO.update(**d['info'])
        exp.NAME = exp.INFO['name']
        exp.SUMMARY.update(**d['summary'])
        update_args(exp.ARGS, d['args'])

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

    exp.configure_from_yaml(config_file=args.config_file)

    command_line_args = dict(data={}, model={}, optimizer={}, train={})
    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
            elif k in model.kwargs:
                head = 'model'
                tail = k
            else:
                continue
            command_line_args[head][tail] = v

    def update_nested_dicts(from_d, to_d):
        for k, v in from_d.items():
            if (k in to_d) and isinstance(to_d[k], dict):
                if not isinstance(v, dict):
                    raise ValueError('Updating dict entry with non-dict.')
                update_nested_dicts(v, to_d[k])
            else:
                to_d[k] = v

    update_nested_dicts(command_line_args['model'], model.kwargs)
    command_line_args['model'].update(**model.kwargs)
    update_args(command_line_args, exp.ARGS)

    for k, v in exp.ARGS.items():
        logger.info('Ultimate {} arguments: \n{}'
                    .format(k, pprint.pformat(v)))

    return model, reload_nets
