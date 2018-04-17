'''Module for holding user-defined configurations.

This should be edited through the config.yaml file

'''

import logging
import yaml


logger = logging.getLogger('cortex.config')

VIZ = {}
TV_PATH = None
DATA_PATHS = {}
OUT_PATH = None
ARCH_PATHS = {}


def update_config(config_file):
    global VIZ, TV_PATH, DATA_PATHS, OUT_PATH
    if config_file is not None:
        logger.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as f:
            d = yaml.load(f)

            viz = d.get('viz', {})
            VIZ.update(**viz)

            torchvision_data_path = d.get('torchvision_data_path', None)
            data_paths = d.get('data_paths', {})
            TV_PATH = torchvision_data_path
            DATA_PATHS.update(**data_paths)
            arch_paths = d.get('arch_paths', {})
            ARCH_PATHS.update(**arch_paths)
            OUT_PATH = d.get('out_path', None)

            logger.debug('User-defined configs: {}'.format(d))
