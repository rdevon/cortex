'''Builds arch

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

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


class ModelHandler(Handler):
    '''
    Simple dict-like container for nn.Module's
    '''

    _type = nn.Module
    _get_error_string = 'Model `{}` not found. You must add it in `build_models` (as a dict entry). Found: {}'
    _special = Handler()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #super().unsafe_set('special', Handler())

    def check_key_value(self, k, v):
        if k in self:
            logger.warning('Key {} already in MODEL_HANDLER, ignoring.'.format(k))
            return False

        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if isinstance(v, (list, tuple)):
            for v_ in v:
                self.check_key_value(k, v_)
        elif self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. Only `{}` and subclasses (or tuples of {}) are supported'.format(
                type(v), k, self._type, self._type))

        return True

    def get_special(self, key):
        return self._special[key]

    def add_special(self, **kwargs):
        self._special.update(**kwargs)

MODEL_HANDLER = ModelHandler()


class ArchHandler(object):
    def __init__(self, defaults=None, setup=None, build=None, Dataset=None, DataLoader=None, transform=None,
                 train_routines=None, test_routines=None, finish_train_routines=None, finish_test_routines=None):
        self.defaults = defaults
        self.setup = setup
        self.build = build
        self.Dataset = Dataset
        self.DataLoader = DataLoader
        self.transform = transform
        self.train_routines = train_routines
        self.test_routines = test_routines or {}
        self.finish_train_routines = finish_train_routines or {}
        self.finish_test_routines = finish_test_routines or {}


def setup_arch(arch):
    global ARCH
    logger.info('Using architecture `{}`'.format(arch))
    ARCH = ARCHS.get(arch, None)
    if ARCH is None:
        raise ValueError('Arch not found ({}). Did you register it? '
                         'Available: {}'.format(
            arch, ARCHS.keys()))
    return ARCH


_arch_keys_required = dict(
    DEFAULT_CONFIG='defaults',
    TRAIN_ROUTINES='train_routines',
    BUILD='build'
)

_arch_keys_optional = dict(
    TEST_ROUTINES='test_routines',
    FINISH_TRAIN_ROUTINES='finish_train_routines',
    FINISH_TEST_ROUTINES='finish_test_routines',
    SETUP='setup',
    Dataset='Dataset',
    DataLoader='DataLoader',
    transform='transform'
)


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

            arch_dict = {}
            for k, v in _arch_keys_required.items():
                if not hasattr(m, k):
                    logger.warning('Architecture (module) {} lacks `{}` skipping'.format(fnp, k))
                    success = False
                else:
                    arch_dict[v] = getattr(m, k)

            for k, v in _arch_keys_optional.items():
                if hasattr(m, k):
                    arch_dict[v] = getattr(m, k)
                else:
                    arch_dict[v] = None

            if name in ARCHS.keys():
                logger.warning('Architecture (module) {} has the same name '
                               '(and path structure) as another architecture, skipping'.format(name))
                success = False

            if success:
                namep = name + '.' + fn[:-3]
                m.logger = logging.getLogger('cortex.arch' + namep)
                ARCHS[namep] = ArchHandler(**arch_dict)
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
    global ARCH, MODEL_HANDLER

    logger.debug('Model args: {}'.format(model_args))
    ARCH.build(data_handler, MODEL_HANDLER, **model_args)

    if ARCH.test_routines is not None:
        for k in ARCH.train_routines:
            if not k in ARCH.test_routines:
                ARCH.test_routines[k] = ARCH.train_routines[k]
    else:
        ARCH.test_routines = ARCH.train_routines


def reload_models(**reload_models):
    global MODEL_HANDLER
    if MODEL_HANDLER is None:
        raise RuntimeError('MODEL_HANDLER not set. `reload_models` should only be used after '
                           '`models.setup_models` has been called.')
    for k, v in reload_models.items():
        logger.info('Reloading model {}'.format(k))
        logger.debug(v)
        MODEL_HANDLER[k] = v
