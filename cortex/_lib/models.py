'''Builds arch

'''

import importlib
import logging
import os
import sys
import time

from . import data, optimizer
from .parsing import parse_docstring, parse_inputs, parse_kwargs
from .handlers import (aliased, prefixed, NetworkHandler, LossHandler,
                       ResultsHandler)
from .utils import bad_values, update_dict_of_lists
from .viz import VizHandler


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.models')

_ROUTINE_PLUGINS = {}
_BUILD_PLUGINS = {}
MODEL_PLUGINS = {}


def check_plugin(plugin, plugin_type_str, D):
    if plugin.plugin_name is None:
        ValueError('Set `plugin_name` static member for plugin.')
    if plugin.plugin_name in D:
        raise KeyError(
            'plugin_name `{}` already registered as a {} plugin in cortex. '
            'Try using another one.'.format(
                plugin_type_str, plugin.plugin_name))

    for k in plugin._protected:
        if hasattr(plugin, k):
            raise AttributeError('{} is a protected attribute.'.format(k))

    for k in plugin._required:
        v = getattr(plugin, k, None)
        if v is None:
            raise AttributeError(
                'Plugin must have {} attribute set.'.format(k))
        else:
            setattr(plugin, k, v)


def register_model(plugin):
    global MODEL_PLUGINS

    if plugin.__name__ in MODEL_PLUGINS:
        raise KeyError('{} already registered under the same name.'
                       .format(plugin.__name__))

    MODEL_PLUGINS[plugin.__name__] = plugin


class PluginType(type):
    def __new__(cls, name, bases, attrs):
        help = {}
        kwargs = {}
        args = set()

        for key in ['build', 'routine', 'visualize', 'train_step',
                    'eval_step']:
            if key in attrs:
                attr = attrs[key]
                help_ = parse_docstring(attr)
                kwargs_ = parse_kwargs(attr)
                args_ = set(parse_inputs(attr))

                for k, v in help_.items():
                    if k in help and v != help[k]:
                        cls._warn_inconsitent_help(key, k, v, kwargs[k])

                for k, v in kwargs_.items():
                    if k in kwargs and v != kwargs[k]:
                        cls._warn_inconsitent_kwargs(key, k, v, kwargs[k])

                help.update(**help_)
                kwargs.update(**kwargs_)
                args |= args_

        attrs['_help'] = help
        attrs['_kwargs'] = kwargs
        attrs['_args'] = args

        return super(PluginType, cls).__new__(cls, name, bases, attrs)

    def _warn_inconsitent_help(cls, k, v, v_):
        logger.warning('Inconsistent docstring found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))

    def _warn_inconsitent_kwargs(cls, k, v, v_):
        logger.warning('Inconsistent keyword defaults found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))


class ModelPluginBase(metaclass=PluginType):
    _viz = VizHandler()
    _data = data.DATA_HANDLER
    _optimizers = optimizer.OPTIMIZERS

    _kwargs = dict()
    _help = dict()
    _owners = dict()
    _training_nets = dict()

    _all_nets = NetworkHandler(allow_overwrite=False)
    _all_losses = LossHandler(_all_nets)
    _all_results = ResultsHandler()

    _all_epoch_results = ResultsHandler()
    _all_epoch_losses = ResultsHandler()
    _all_epoch_times = ResultsHandler()

    def __init__(self, contract=None):
        self._contract = None
        self._kwarg_dict = dict()
        self._input_dict = dict()

        if contract:
            contract = self._check_contract(contract)
            self._accept_contract(contract)

        if self._contract and len(self._contract['nets']) > 0:
            self._nets = aliased(self._all_nets, aliases=contract['nets'])
            self._losses = aliased(self._all_losses, aliases=contract['nets'])
        else:
            self._nets = aliased(self._all_nets)
            self._losses = aliased(self._all_losses)

        try:
            self.eval()
        except NotImplementedError:
            self.eval = self.routine

        for k in ['build', 'routine', 'visualize', 'train_step',
                  'eval_step']:
            fn = getattr(self, k)
            fid = self._get_id(fn)
            self._owners[fid] = self.__class__.__name__

        self._wrap_build()
        self._wrap_routine()
        self.train_step = self._wrap_step(self.train_step)
        self.eval_step = self._wrap_step(self.eval_step)

        self._results = prefixed(
            self._all_results, prefix=self.__class__.__name__)
        self._epoch_results = prefixed(
            self._all_epoch_results, prefix=self.__class__.__name__)
        self._epoch_losses = prefixed(
            self._all_epoch_losses, prefix=self.__class__.__name__)
        self._epoch_times = prefixed(
            self._all_epoch_times, prefix=self.__class__.__name__)

    def _reset_epoch(self):
        self._all_epoch_results.clear()
        self._all_epoch_losses.clear()
        self._all_epoch_times.clear()

    def _get_id(self, fn):
        return fn

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def args(self):
        return self._args

    @property
    def help(self):
        return self._help

    @property
    def results(self):
        return self._results

    @property
    def epoch_results(self):
        return self._epoch_results

    @property
    def epoch_losses(self):
        return self._epoch_losses

    @property
    def epoch_times(self):
        return self._epoch_times

    @property
    def nets(self):
        return self._nets

    @property
    def losses(self):
        return self._losses

    @property
    def viz(self):
        return self._viz

    @property
    def data(self):
        return self._data

    def __setattr__(self, key, value):
        if isinstance(value, ModelPluginBase):
            model = value
            kwargs = model.kwargs
            help = model.help
            if model._contract:
                kwargs = dict((model._contract['kwargs'].get(k, k), v)
                              for k, v in kwargs.items())
                help = dict((model._contract['kwargs'].get(k, k), v)
                            for k, v in help.items())
            for k, v in kwargs.items():
                if k not in self.kwargs:
                    self.kwargs[k] = v
                if k not in self.help:
                    self.help[k] = help[k]

        super().__setattr__(key, value)

    def _check_contract(self, contract):
        kwargs = contract.pop('kwargs', {})
        nets = contract.pop('nets', {})
        inputs = contract.pop('inputs', {})

        if len(contract) > 0:
            raise KeyError('Unknown keys in contract: {}'
                           .format(tuple(contract.keys())))

        for k, v in kwargs.items():
            if k not in self.kwargs:
                raise KeyError('Invalid contract: {} does not have any '
                               'arguments called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        for k, v in inputs.items():
            if k not in self.args:
                raise KeyError('Invalid contract: {} does not have any '
                               'inputs called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        return dict(inputs=inputs, kwargs=kwargs, nets=nets)

    def _accept_contract(self, contract):
        if self._contract is not None:
            raise ValueError('Cannot accept more than one contract.')

        self._contract = contract

        for k in ['build', 'routine', 'visualize', 'train_step',
                  'eval_step']:
            fn = getattr(self, k)
            fid = self._get_id(fn)
            self._kwarg_dict[fid] = contract['kwargs']
            self._input_dict[fid] = contract['inputs']

    def _wrap_build(self):
        fn = self.build

        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped._fn = fn
        self.build = wrapped

    def _wrap_routine(self):
        '''

        Set to `requires_grad` for models that are trained with this routine.

        Args:
            routine:

        Returns:

        '''

        fn = self.routine

        def wrapped(*args, **kwargs):
            fid = self._get_id(fn)

            if fid not in self._training_nets:
                losses_before = dict(kv for kv in self.losses.items())
                fn(*args, **kwargs)
                losses_after = dict(kv for kv in self.losses.items())

                training_nets = []

                for k, v in losses_after.items():
                    if k not in losses_before:
                        training_nets.append(k)
                    elif v != losses_before[k]:
                        training_nets.append(k)
                self._training_nets[fid] = training_nets
            else:
                training_nets = self._training_nets[fid]

            for k in training_nets:
                net = self.nets[k]
                self._optimizers[k].zero_grad()
                for p in net.parameters():
                    p.requires_grad = k in training_nets
                net.train()

            start = time.time()
            output = fn(*args, **kwargs)
            self._check_bad_values()
            end = time.time()

            owner = self._owners[self._get_id(fn)]
            update_dict_of_lists(self.epoch_results, **self.results)
            update_dict_of_lists(self.epoch_times, **{owner: end - start})
            update_dict_of_lists(self.epoch_losses, **self.losses)

            return output

        wrapped._fn = fn
        self.routine = wrapped

    def _wrap_step(self, fn):

        def wrapped(*args, **kwargs):
            for net in self.nets.values():
                net.eval()

            self._all_losses.clear()
            self._all_results.clear()

            output = fn(*args, **kwargs)

            return output

        wrapped._fn = fn
        return wrapped

    def get_inputs(self, fn):
        fid = self._get_id(fn._fn)
        input_dict = self._input_dict.get(fid, {})
        input_keys = parse_inputs(fn._fn)

        inputs = []
        for k in input_keys:
            key = input_dict.get(k, k)
            inp = self.data[key]
            inputs.append(inp)

        return inputs

    def get_kwargs(self, fn):
        fid = self._get_id(fn._fn)
        kwarg_dict = self._kwarg_dict.get(fid, {})
        kwarg_dict = dict((v, k) for k, v in kwarg_dict.items())
        kwarg_keys = parse_kwargs(fn._fn).keys()

        kwargs = dict()
        for k in kwarg_keys:
            key = kwarg_dict.get(k, k)
            value = self.kwargs[key]
            kwargs[k] = value

        return kwargs

    def _get_training_nets(self):
        training_nets = []
        for v in self._training_nets.values():
            training_nets += v

    def _check_bad_values(self):
        # Check for bad numbers
        bads = bad_values(self.results)
        if bads:
            print(
                'Bad values found (quitting): {} \n All:{}'.format(
                    bads, self.results))
            exit(0)

        bads = bad_values(self.losses)
        if bads:
            print(
                'Bad values found (quitting): {} \n All:{}'.format(
                    bads, self.losses))
            exit(0)


def setup_model(model_key):
    global MODEL
    logger.info('Using model `{}`'.format(model_key))
    MODEL = MODEL_PLUGINS[model_key]
    return MODEL


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


def import_directory(p, name):
    '''
    Adds custom directories to the framwework
    '''

    global ARCHS

    if p.endswith('/'):
        p = p[:-1]

    logger.info('Adding {} to `sys.path`.'.format(p))
    sys.path.append(p)

    for fn in os.listdir(p):
        if fn.endswith('.py') and fn not in _ignore:
            fnp = fn[:-3]
            importlib.import_module(fnp)
            try:
                importlib.import_module(fnp)
            except Exception as e:
                logger.warning(
                    'Import of architecture (module) {} failed ({})'.format(
                        fnp, e))

        '''
        elif os.path.isdir(fn):
            if fn.endswith('/'):
                fn = fn[:-1]
            if fn not in _ignore:
                add_directory(fn, name + '.' + os.path.basename(fn))
        '''


def find_models(model_paths):
    for k, p in model_paths.items():
        import_directory(p, k)

    global MODEL_PLUGINS
    keys = list(MODEL_PLUGINS.keys())
    for k in keys:
        v = MODEL_PLUGINS[k]
        v.check()
        try:
            v.check()
        except Exception as e:
            logger.warning('`{}` checks failed ({}).'.format(k, e))
            MODEL_PLUGINS.pop(k)


def build_networks():
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''
    for build_key, build in MODEL.builds.items():
        logger.debug('{} build args: {}'.format(build_key, build.kwargs))
        build()
