'''Builds models

'''

import logging

import classifier, gan


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


def build_model(data_dims, **model_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''

    logger.debug('Model args: {}'.format(model_args))
    if hasattr(ARCH, 'GLOBALS'):
        for k, v in ARCH.GLOBALS.items():
            if k.lower() in data_dims.keys():
                v_ = data_dims.pop(k.lower())
                if v_ != v:
                    logger.warning('Changing {} to {} from default {}'.format(
                        k, v_, v))
                    v = v_
            logger.debug('Setting module global {} to {}'.format(k, v))
            setattr(ARCH, k, v)

    if hasattr(ARCH, 'build_model'):
        return getattr(ARCH, 'build_model')(**model_args)
    else:
        raise NotImplementedError('Module lacks `build_model` method')

_archs = dict(classifier=classifier, gan=gan)