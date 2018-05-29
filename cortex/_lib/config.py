'''Module for config

'''

import logging
from os import path
import pprint
import yaml

from .utils import Handler


logger = logging.getLogger('cortex.config')


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
            data_paths = d.get('data_paths', {})
            arch_paths = d.get('arch_paths', {})
            out_path = d.get('out_path', None)

            CONFIG.update(viz=viz,
                          data_paths=data_paths, arch_paths=arch_paths, out_path=out_path)
    else:
        logger.warning('config.yaml not found')