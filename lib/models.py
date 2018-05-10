'''Builds arch

'''

import importlib
import logging
import sys
import os

from .config import ARCH_PATHS
from .data import DATA_HANDLER
from . import exp


logger = logging.getLogger('cortex.models')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arch_dir = os.path.abspath(os.path.join(root, 'arch'))
ARCH_PATHS.update(core=arch_dir)
ARCHS = dict()
ARCH = None


def add_directory(p, name):
    global ARCHS

    if p.endswith('/'):
        p = p[:-1]

    sys.path.append(p)

    for fn in os.listdir(p):
        if fn.endswith('.py'):
            fnp = os.path.basename(p) + '.' + fn[:-3]
            logger.info('Attempting to import {}'.format(fnp))
            try:
                m = importlib.import_module(fnp)
                success = True
            except Exception as e:
                logger.warning('Import of architecture (module) {} failed ({})'.format(fnp, type(e)))
                success = False
            if not success:
                pass
            elif not hasattr(m, 'DEFAULT_CONFIG'):
                logger.warning('Architecture (module) {} lacks `DEFAULT_CONFIG` dictionary, skipping'.format(fnp))
            elif not hasattr(m, 'build_model'):
                logger.warning('Architecture (module) {} lacks `build_model` method, skipping'.format(fnp))
            elif not hasattr(m, 'ROUTINES'):
                logger.warning('Architecture (module) {} lacks `ROUTINES` dictionary, skipping'.format(fnp))
            elif name in ARCHS.keys():
                logger.warning('Architecture (module) {} has the same name '
                               '(and path structure) as another architecture, skipping'.format(name))
            else:
                namep = name + '.' + fn[:-3]
                m.logger = logging.getLogger('cortex.arch' + namep)
                ARCHS[namep] = m
        elif os.path.isdir(fn):
            if fn.endswith('/'):
                fn = fn[:-1]
            add_directory(fn, name + '.' + os.path.basename(fn))

for k, p in ARCH_PATHS.items():
    add_directory(p, k)


def setup(arch):
    global ARCH
    logger.info('Using architecture `{}`'.format(arch))
    ARCH = ARCHS.get(arch, None)
    if ARCH is None:
        raise ValueError('Arch not found ({}). Did you register it? '
                         'Available: {}'.format(
            arch, ARCHS.keys()))
    return ARCH


def setup_model(**model_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''

    models = dict()
    logger.debug('Model args: {}'.format(model_args))

    getattr(ARCH, 'build_model')(DATA_HANDLER, models, **model_args)

    exp.setup(models, ARCH.ROUTINES)