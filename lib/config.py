'''Module for holding user-defined configurations.

This should be edited through the config.yaml file

'''

import logging
import yaml


logger = logging.getLogger('cortex.config')

VIZ = {}
TV_PATH = None
LOCAL_PATH = None
DATA_PATHS = {}
OUT_PATH = None
ARCH_PATHS = {}


def update_config(config_file):
    global VIZ, TV_PATH, DATA_PATHS, OUT_PATH, LOCAL_PATH
    if config_file is not None:
        logger.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as f:
            d = yaml.load(f)

            viz = d.get('viz', {})
            VIZ.update(**viz)

            TV_PATH = d.get('torchvision_data_path', None)
            data_paths = d.get('data_paths', {})
            DATA_PATHS.update(**data_paths)
            arch_paths = d.get('arch_paths', {})
            ARCH_PATHS.update(**arch_paths)
            OUT_PATH = d.get('out_path', None)
            LOCAL_PATH = d.get('local_data_path', None)

            logger.debug('User-defined configs: {}'.format(d))
