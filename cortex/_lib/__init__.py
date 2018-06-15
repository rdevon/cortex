'''Cortex setup

'''

import copy
import logging
import pprint

from . import config, exp, log_utils, models
from .parsing import default_args, parse_args
from .viz import init as viz_init

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.init')


def setup_cortex():
    '''Sets up cortex

    Finds all the models in cortex, parses the command line, and sets the
    logger.

    Returns:
        TODO

    '''
    models.find_models(config.CONFIG.arch_paths)

    args = parse_args(models.MODEL_PLUGINS)

    log_utils.set_stream_logger(args.verbosity)

    return args


def setup_experiment(args):
    '''Sets up the experiment

    Args:
        args: TODO

    '''
    exp.setup_device(args.device)
    model_name = args.command

    experiment_args = copy.deepcopy(parsing.default_args)
    exp.update_args(experiment_args)

    model = models.setup_model(model_name)

    viz_init(config.CONFIG.viz)

    if args.reload and not args.load_models:
        exp.reload(args.reload, args.reloads, args.name,
                   args.out_path, args.clean, config.CONFIG)
    else:
        name = args.name or model_name
        exp.setup_new(model.defaults, name, args.out_path, args.clean,
                      config.CONFIG, args.load_models, args.reloads)

    exp.configure_from_yaml(config_file=args.config_file)

    command_line_args = dict(data={}, builds={}, routines={},
                             optimizer={}, train={})
    model_args = model.unpack_args()
    command_line_args.update(**model_args)

    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
                command_line_args[head][tail] = v

    exp.update_args(command_line_args)

    for k, v in exp.ARGS.items():
        logger.info('Ultimate {} arguments: \n{}'
                    .format(k, pprint.pformat(v)))

    model.kwargs.update(**exp.ARGS['routines'])

    if model.setup is not None:
        model.setup(**exp.ARGS)
