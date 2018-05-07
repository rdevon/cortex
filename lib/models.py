'''Builds arch

'''

import importlib
import logging
import sys
import os

import torch.nn as nn

from .config import ARCH_PATHS
from .utils import Handler


logger = logging.getLogger('cortex.models')

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
arch_dir = os.path.abspath(os.path.join(root, 'arch'))
ARCH_PATHS.update(core=arch_dir)
ARCHS = dict()
ARCH = None


class ModelHandler(Handler):
    '''
    Simple dict-like container for nn.Module's
    '''

    _type = nn.Module
    _get_error_string = 'Model `{}` not found. You must add it in `build_models` (as a dict entry). Found: {}'

    def check_key_value(self, k, v):
        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if isinstance(v, (list, tuple)):
            for v_ in v:
                self.check_key_value(k, v_)
        elif self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. Only `{}` and subclasses (or tuples of {}) are supported'.format(
                type(v), k, self._type, self._type))


def setup(arch):
    global ARCH
    logger.info('Using architecture `{}`'.format(arch))
    ARCH = ARCHS.get(arch, None)
    if ARCH is None:
        raise ValueError('Arch not found ({}). Did you register it? '
                         'Available: {}'.format(
            arch, ARCHS.keys()))
    return ARCH


def add_directory(p, name):
    '''
    Adds custom directories to the framwework
    '''

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
            elif not hasattr(m, 'BUILD'):
                logger.warning('Architecture (module) {} lacks `BUILD` method, skipping'.format(fnp))
            elif not hasattr(m, 'TRAIN_ROUTINES'):
                logger.warning('Architecture (module) {} lacks `TRAIN_ROUTINES` dictionary, skipping'.format(fnp))
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


def setup_model(data_handler, **model_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''

    models = ModelHandler()
    logger.debug('Model args: {}'.format(model_args))

    getattr(ARCH, 'BUILD')(data_handler, models, **model_args)
    train_routines = ARCH.TRAIN_ROUTINES
    if hasattr(ARCH, 'TEST_ROUTINES'):
        test_routines = ARCH.TEST_ROUTINES
        for k in train_routines:
            if not k in test_routines:
                test_routines[k] = train_routines[k]
    else:
        test_routines = train_routines

    finish_train_routines = getattr(ARCH, 'FINISH_TRAIN_ROUTINES', {})
    finish_test_routines = getattr(ARCH, 'FINISH_TEST_ROUTINES', {})

    routines = dict(train_routines=train_routines, test_routines=test_routines,
                    finish_train_routines=finish_train_routines, finish_test_routines=finish_test_routines)

    return models, routines