'''Builds models

'''

import importlib
import logging
import os

from . import classifier, gan, discrete_gan, featnet


arch_names = ['ali', 'gan', 'discrete_gan', 'classifier', 'copulas', 'embedding_gan', 'featnet', 'minet', 'nat', 'vae', 'vral']
logger = logging.getLogger('cortex.models')

ARCHS = dict()

mod_path = os.path.dirname(__file__)
for arch_name in arch_names:
    m = importlib.import_module('models.' + arch_name)
    ARCHS[arch_name] = m

ARCH = None


def setup(arch):
    global ARCH
    logger.info('Using architecture `{}`'.format(arch))
    ARCH = ARCHS.get(arch, None)
    if ARCH is None:
        raise ValueError('Arch not found ({}). Did you register it? '
                         'Available: {}'.format(
            arch, ARCHS.keys()))

    return ARCH


def build_model(data_handler, **model_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''

    logger.debug('Model args: {}'.format(model_args))

    if hasattr(ARCH, 'build_model'):
        return getattr(ARCH, 'build_model')(data_handler, **model_args)
    else:
        raise NotImplementedError('Module lacks `build_model` method')


def register(key, mod):
    ARCHS[key] = mod