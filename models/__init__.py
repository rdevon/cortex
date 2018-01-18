'''Builds models

'''

import logging

import classifier, gan, discrete_gan, featnet


logger = logging.getLogger('cortex.models')

ARCH = None


def setup(arch):
    global ARCH
    logger.info('Using architecture `{}`'.format(arch))
    ARCH = _archs.get(arch, None)
    if ARCH is None:
        raise ValueError('Arch not found (``). Did you register it? '
                         'Available: {}'.format(
            arch, _archs.keys()))

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

_archs = dict(classifier=classifier, discrete_gan=discrete_gan, gan=gan, featnet=featnet)