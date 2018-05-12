'''Builds arch

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import importlib
import inspect
import logging
import sys
import os
import traceback

import torch.nn as nn

from .utils import Handler


logger = logging.getLogger('cortex.models')
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
                 train_routines=None, test_routines=None, finish_train_routines=None, finish_test_routines=None,
                 doc=None, kwargs=dict(), signatures=[], info=dict()):
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

        self.doc = doc
        self.kwargs = kwargs
        self.signatures = signatures
        self.info = info

    def unpack_args(self, args):
        model = Handler()
        routines = Handler()
        for k, v in vars(args).items():
            for sig_k, sig_v in self.signatures.items():
                if k in sig_v:
                    if sig_k == 'model':
                        model[k] = v
                    else:
                        if sig_k not in routines:
                            routines[sig_k] = {k:v}
                        else:
                            routines[sig_k][k] = v

        return Handler(model=model, routines=routines)


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

_ignore = ['__init__.py', '__pycache__']


def add_directory(p, name):
    '''
    Adds custom directories to the framwework
    '''

    global ARCHS

    if p.endswith('/'):
        p = p[:-1]

    logger.info('Adding {} to `sys.path`.'.format(p))
    sys.path.append(p)

    for fn in os.listdir(p):
        if fn.endswith('.py') and not fn in _ignore:
            fnp = fn[:-3]
            logger.info('Importing {}'.format(fnp))
            try:
                success = True
                m = importlib.import_module(fnp)

                arch_dict = {}
                for k, v in _arch_keys_required.items():
                    if success:
                        if not hasattr(m, k):
                            raise AttributeError('Architecture (module) {} lacks `{}` skipping'.format(fnp, k))
                        else:
                            arch_dict[v] = getattr(m, k)

                kwargs = {}
                signatures = {}
                for k, v in arch_dict['train_routines'].items():
                    sig = inspect.signature(v)
                    signatures[k] = []
                    for i, (sk, sv) in enumerate(sig.parameters.items()):
                        if i < 5 and (sk == 'kwargs' or sv.default != inspect._empty):
                            raise ValueError('First 5 elements of routines ({}) must be parameters'.format(k))
                        elif i >= 5 and (sk != 'kwargs' and sv.default == inspect._empty):
                            raise ValueError('Only the first 5 elements of routines ({}) can be parameters'.format(k))
                        elif i >= 5:
                            if sk == 'kwargs':
                                pass # For handling old-style files
                            else:
                                signatures[k].append(sv.name)
                                if sv.default is not None:
                                    if sv.name in kwargs and kwargs[sv.name] != sv.default:
                                        logger.warning('Multiple values found for {}. This may be undesired.'.format(sv.name))
                                    kwargs[sv.name] = sv.default

                sig = inspect.signature(arch_dict['build'])
                signatures['model'] = []
                for i, (sk, sv) in enumerate(sig.parameters.items()):
                    if i < 2 and (sk == 'kwargs' or sv.default != inspect._empty):
                        raise ValueError('First 2 elements of BUILD must be parameters ({})'.format(sk))
                    elif i >= 2 and (sk != 'kwargs' and sv.default == inspect._empty):
                        raise ValueError('Only the first 2 elements of BUILD can be parameters ({})'.format(sk))
                    elif i >= 2:
                        if sk == 'kwargs':
                            pass
                        signatures['model'].append(sv.name)
                        if sv.default is not None:
                            if sv.name in kwargs and kwargs[sv.name] != sv.default:
                                logger.warning('Multiple values found for {}. This may be undesired.'.format(sv.name))
                            kwargs[sv.name] = sv.default
                arch_dict['kwargs'] = kwargs
                arch_dict['signatures'] = signatures

                for k, v in _arch_keys_optional.items():
                    if hasattr(m, k):
                        arch_dict[v] = getattr(m, k)
                    else:
                        arch_dict[v] = None

                if hasattr(m, 'INFO'):
                    info = getattr(m, 'INFO')
                    arch_dict['info'] = info

                arch_dict['doc'] = m.__doc__

                if name in ARCHS.keys():
                    logger.warning('Architecture (module) {} has the same name '
                                   '(and path structure) as another architecture, skipping'.format(name))
                    success = False

            except Exception as e:
                logger.warning('Import of architecture (module) {} failed ({})'.format(fnp, e))
                success = False

            finally:
                if success:
                    namep = name + '.' + fn[:-3]
                    m.logger = logging.getLogger('cortex.arch' + namep)
                    ARCHS[namep] = ArchHandler(**arch_dict)
        elif os.path.isdir(fn):
            if fn.endswith('/'):
                fn = fn[:-1]
            if fn not in _ignore:
                add_directory(fn, name + '.' + os.path.basename(fn))


def find_archs(arch_paths):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch_dir = os.path.abspath(os.path.join(root, 'arch'))
    arch_paths.update(core=arch_dir)
    for k, p in arch_paths.items():
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
