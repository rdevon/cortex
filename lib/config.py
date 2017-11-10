'''Module for holding user-defined configurations.

This should be edited through the config.yaml file

'''

import logging
import yaml


logger = logging.getLogger('cortex.config')

VIZ = {}
DATA_PATH = None
OUT_PATH = None


def update_config(config_file):
    global VIZ, DATA_PATH, OUT_PATH
    if config_file is not None:
        logger.debug('Open config file {}'.format(config_file))
        with open(config_file, 'r') as f:
            d = yaml.load(f)

            viz = d.get('viz', {})
            VIZ.update(**viz)

            data_path = d.get('data_path', None)
            if data_path is None:
                raise ValueError('`data_path` not set in config.yaml')
            DATA_PATH = data_path

            OUT_PATH = d.get('out_path', None)

            logger.debug('User-defined configs: {}'.format(d))
