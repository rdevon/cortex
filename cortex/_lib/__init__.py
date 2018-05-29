'''Cortex setup

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import copy
import logging
import pprint

from . import config, exp, log_utils, models
from .parsing import _args as default_args, parse_args
from .utils import Handler
from .viz import init as viz_init


logger = logging.getLogger('cortex.init')


def setup_cortex():
    config.set_config()
    models.find_archs(config.CONFIG.arch_paths)

    args = parse_args(models.ARCHS)
    experiment_args = copy.deepcopy(default_args)
    exp.update_args(experiment_args)

    log_utils.set_stream_logger(args.verbosity)

    exp.setup_device(args.device)
    models.setup_arch(args.arch)
    viz_init(config.CONFIG.viz)

    if args.reload and not args.load_models:
        exp.reload(args.reload, args.reloads, args.name, args.out_path, args.clean, config.CONFIG)
    else:
        name = args.name or args.arch
        exp.setup_new(models.ARCH.defaults, name, args.out_path, args.clean, config.CONFIG,
                      args.load_models, args.reloads)

    exp.configure_from_yaml(config_file=args.config_file)

    command_line_args = dict(data={}, model={}, routines={}, optimizer={}, train={})
    arch_args = models.ARCH.unpack_args(args)
    command_line_args.update(**arch_args)

    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
                command_line_args[head][tail] = v

    exp.update_args(command_line_args)
    exp.copy_test_routines()

    for k, v in exp.ARGS.items():
        logger.info('Ultimate {} arguments: \n{}'.format(k, pprint.pformat(v)))

    exp.ARGS.data.copy_to_local = args.copy_to_local

    if models.ARCH.setup is not None:
        models.ARCH.setup(**exp.ARGS)