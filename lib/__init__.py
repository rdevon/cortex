'''Cortex setup

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import copy
import logging
from os import path
import pprint
import yaml

from . import data, exp, log_utils, models
from .parsing import _args as default_args, parse_args
from .utils import Handler
from .viz import init as viz_init


logger = logging.getLogger('cortex.init')


CONFIG = Handler()
def set_config():
    global CONFIG
    config_file = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'config.yaml')
    if not path.isfile(config_file):
        config_file = None

    if config_file is not None:
        logger.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as f:
            d = yaml.load(f)
            logger.debug('User-defined configs: {}'.format(pprint.pformat(d)))

            viz = d.get('viz', {})
            torchvision_data_path = d.get('torchvision_data_path', None)
            toy_data_path = d.get('toy_data_path', None)
            data_paths = d.get('data_paths', {})
            arch_paths = d.get('arch_paths', {})
            out_path = d.get('out_path', None)
            local_data_path = d.get('local_data_path', None)

            CONFIG.update(viz=viz, torchvision_data_path=torchvision_data_path, toy_data_path=toy_data_path,
                          data_paths=data_paths, arch_paths=arch_paths, out_path=out_path,
                          local_data_path=local_data_path)
    else:
        logger.warning('config.yaml not found')


def setup_cortex():
    set_config()
    data.set_config(CONFIG)
    models.find_archs(CONFIG.arch_paths)

    args = parse_args(models.ARCHS)
    experiment_args = copy.deepcopy(default_args)
    exp.update_args(experiment_args)

    log_utils.set_stream_logger(args.verbosity)

    exp.setup_device(args.device)
    models.setup_arch(args.arch)
    viz_init(CONFIG.viz)

    if args.reload:
        exp.reload(args.reload, args.reloads, args.name, args.out_path, args.clean, CONFIG)
    else:
        name = args.name or args.arch
        exp.setup_new(models.ARCH.defaults, name, args.out_path, args.clean, CONFIG)

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
    exp.ARGS.train.test_mode = args.test

    if models.ARCH.setup is not None:
        models.ARCH.setup(**exp.ARGS)