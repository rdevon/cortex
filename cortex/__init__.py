"""
TODO
"""
import copy
import logging
import pprint
from os import path
import yaml
from . import data, experiment, models
from cortex.utils import log_utils
from cortex.utils.parsing import _args as default_args, parse_args
from .utils import Handler
from cortex import init as viz_init
from ._version import get_versions
from .utils._appdirs import AppDirs
VERSIONS = get_versions()

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'
__descr__ = 'Distributed Asynchronous [black-box] Optimization'
__version__ = VERSIONS['version']
__license__ = 'BSD-3-Clause'
__author__ = 'Cortex Team - MILA, Université de Montréal'
__author_short__ = 'MILA'
__author_email__ = 'lisa_labo@iro.umontreal.ca'
__authors__ = {
    'rdevon': ('Devon Hjelm', 'devonhjelm@gmail.com'),
}
__url__ = 'https://github.com/rdevon/cortex2.0'

DIRS = AppDirs(__name__, __author_short__)
LOGGER = logging.getLogger('cortex.init')
CONFIG = Handler()

del AppDirs
del get_versions


def set_config():
    """
    TODO
    """
    print(
        path.join(
            path.dirname(
                path.dirname(
                    path.abspath(__file__))) +
            '/config/',
            'config.yaml'))
    config_file = path.join(
        path.dirname(
            path.dirname(
                path.abspath(__file__))) +
        '/config/',
        'config.yaml')
    if not path.isfile(config_file):
        config_file = None
    if config_file is not None:
        LOGGER.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as file:
            config_yaml = yaml.load(file)
            LOGGER.debug('User-defined configs: {}'.format(pprint.pformat(config_yaml)))
            viz = config_yaml.get('viz', {})
            torchvision_data_path = config_yaml.get('torchvision_data_path', None)
            toy_data_path = config_yaml.get('toy_data_path', None)
            data_paths = config_yaml.get('data_paths', {})
            arch_paths = config_yaml.get('arch_paths', {})
            out_path = config_yaml.get('out_path', None)
            local_data_path = config_yaml.get('local_data_path', None)
            CONFIG.update(
                viz=viz,
                torchvision_data_path=torchvision_data_path,
                toy_data_path=toy_data_path,
                data_paths=data_paths,
                arch_paths=arch_paths,
                out_path=out_path,
                local_data_path=local_data_path)
    else:
        LOGGER.warning('config.yaml not found')


def setup_cortex():
    """
    TODO
    """
    set_config()
    data.set_config(CONFIG)
    models.find_archs(CONFIG.arch_paths)

    args = parse_args(models.ARCHS)
    experiment_args = copy.deepcopy(default_args)
    experiment.update_args(experiment_args)

    log_utils.set_stream_logger(args.verbosity)

    experiment.setup_device(args.device)
    models.setup_arch(args.arch)
    viz_init(CONFIG.viz)

    if args.reload and not args.load_models:
        experiment.reload(
            args.reload,
            args.reloads,
            args.name,
            args.out_path,
            args.clean,
            CONFIG)
    else:
        name = args.name or args.arch
        experiment.setup_new(
            models.ARCH.defaults,
            name,
            args.out_path,
            args.clean,
            CONFIG,
            args.load_models,
            args.reloads)

    experiment.configure_from_yaml(config_file=args.config_file)

    command_line_args = dict(
        data={},
        model={},
        routines={},
        optimizer={},
        train={})
    arch_args = models.ARCH.unpack_args(args)
    command_line_args.update(**arch_args)

    for k, v in vars(args).items():
        if v is not None:
            if '.' in k:
                head, tail = k.split('.')
                command_line_args[head][tail] = v

    experiment.update_args(command_line_args)
    experiment.copy_test_routines()

    for k, v in experiment.ARGS.items():
        LOGGER.info('Ultimate {} arguments: \n{}'.format(k, pprint.pformat(v)))

    experiment.ARGS.data.copy_to_local = args.copy_to_local

    if models.ARCH.setup is not None:
        models.ARCH.setup(**experiment.ARGS)
