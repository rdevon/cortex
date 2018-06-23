'''Builds arch

'''

import importlib
import logging
import os
import sys
import time

from . import data, optimizer
from .parsing import parse_docstring, parse_inputs, parse_kwargs
from .handlers import aliased, NetworkHandler, LossHandler, ResultsHandler
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

        for key in ['build', 'routine', 'visualize']:
            if key in attrs:
                attr = attrs[key]
                help_ = parse_docstring(attr)
                kwargs_ = parse_kwargs(attr)
                args_ = parse_inputs(attr)

                for k, v in help_.items():
                    if k in help and v != help[k]:
                        cls._warn_inconsitent_help(key, k, v, kwargs[k])

                for k, v in kwargs_.items():
                    if k in kwargs and v != kwargs[k]:
                        cls._warn_inconsitent_kwargs(key, k, v, kwargs[k])

                help.update(**help_)
                kwargs.update(**kwargs_)
                args |= set(args_)

        attrs['_help'] = help
        attrs['_kwargs'] = kwargs
        attrs['_args'] = list(args)

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
    _nets = NetworkHandler(allow_overwrite=False)
    _losses = LossHandler(_nets)
    _training_nets = []

    def __init__(self, contract=None):
        self._train_procedures = []
        self._eval_procedures = []
        self._contact = None

        self._results = ResultsHandler(time=dict(), losses=dict())

        if contract:
            contract = self._check_contract(contract)
            self._accept_contract(contract)
        else:
            if hasattr(self, 'build'):
                self.build = self._wrap(self.build)

            if hasattr(self, 'routine'):
                self.routine = self._wrap(self.routine)

            if hasattr(self, 'visualize'):
                self.visualize = self._wrap(self.visualize)

    def __setattr__(self, key, value):
        if isinstance(value, ModelPluginBase):
            model = value
            kwargs = model.kwargs
            help = model.help
            args = model.args
            if model._contact:
                kwargs = dict((model._contact['kwargs'].get(k, k), v)
                              for k, v in kwargs.items())
                help = dict((model._contact['kwargs'].get(k, k), v)
                            for k, v in help.items())
                args = [model._contact['args'].get(k, k) for k in args]
            for k, v in kwargs.items():
                if k not in self.kwargs:
                    self.kwargs[k] = v
                if k not in self.help:
                    self.help[k] = help[k]
                    
            self._args = list(set(self.args) | set(args))

        super().__setattr__(key, value)

    def _check_contract(self, contract):
        kwargs = contract.pop('kwargs', {})
        nets = contract.pop('nets', {})
        args = contract.pop('args', {})

        if len(contract) > 0:
            raise KeyError('Unknown keys in contract: {}'
                           .format(tuple(contract.keys())))

        for k, v in kwargs.items():
            if k not in self.kwargs:
                raise KeyError('{} does not have any arguments called {}'
                               .format(self.__class__.__name__, k))
            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        for k, v in args.items():
            if k not in self.args:
                raise KeyError('{} does not have any inputs called {}'
                               .format(self.__class__.__name__, k))
            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        return dict(args=args, kwargs=kwargs, nets=nets)

    def _accept_contract(self, contract):
        if self._contact is not None:
            raise ValueError('Cannot accept more than one contract.')

        if hasattr(self, 'build'):
            self.build = self._wrap(self.build, kwargs=contract['kwargs'])

        if hasattr(self, 'routine'):
            self.routine = self._wrap(self.routine, kwargs=contract['kwargs'])

        if hasattr(self, 'visualize'):
            self.visualize = self._wrap(self.visualize,
                                        kwargs=contract['kwargs'])

        self._contact = contract
        self._nets = aliased(self._nets, aliases=contract['nets'])

    def _wrap(self, fn, kwargs=None):
        kwargs = kwargs or {}
        kwargs = dict((v, k) for k, v in kwargs.items())

        kwarg_keys = parse_kwargs(fn).keys()

        def wrapped(*args, **kwargs_):
            kwargs_ = dict((kwargs.get(k, k), v) for k, v in kwargs_.items()
                           if kwargs.get(k, k) in kwarg_keys)
            return fn(*args, **kwargs_)

        return wrapped

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
    def setup(self):
        return self._setup

    @property
    def train_procedures(self):
        return self._train_procedures

    @property
    def eval_procedures(self):
        return self._eval_procedures

    @property
    def results(self):
        return self._results

    @property
    def nets(self):
        return self._nets

    @property
    def losses(self):
        return self._losses

    @property
    def viz(self):
        return self._viz

    def train(self, i, quit_on_bad_values=False):
        return self.run_procedure(
            i, quit_on_bad_values=quit_on_bad_values, train=True)

    def run_procedure(self, i, quit_on_bad_values=False, train=False):
        self._data.next()
        self.reset_routines()
        self._vars.clear()
        mode, procedure, updates = self._train_procedures[i]

        for k, v in self._data.batch.items():
            self._vars['data.' + k] = v

        losses = {}

        for key, update in zip(procedure, updates):
            if not train:
                update = 1
            for u in range(update):
                if u > 0:
                    self._data.next()

                routine = self._routines[key]
                routine.reset()

                # Set to `requires_grad` for models that are trained with this
                # routine.
                if train:
                    for k in optimizer.OPTIMIZERS.keys():
                        net = self.nets[k]
                        optimizer.OPTIMIZERS[k].zero_grad()
                        for p in net.parameters():
                            p.requires_grad = k in routine._training_nets

                start_time = time.time()
                losses_before = {k: v for k, v in self._losses.items()}
                routine()
                losses_after = {k: v for k, v in self._losses.items()}

                # Check which networks the routine changed losses to.
                for k, v in losses_after.items():
                    if k not in routine._training_nets:
                        if k not in losses_before:
                            routine._training_nets.append(k)
                        elif v != losses_before[k]:
                            routine._training_nets.append(k)

                # Backprop the losses.
                keys = list(self._losses.keys())
                for i, k in enumerate(keys):
                    loss = self._losses.pop(k)
                    if k in losses:
                        losses[k] += loss.item()
                    else:
                        losses[k] = loss.item()
                    if train and loss is not None:
                        loss.backward()
                        optimizer.OPTIMIZERS[k].step()

                end_time = time.time()

                # Check for bad numbers
                bads = bad_values(routine.results)
                if bads and quit_on_bad_values:
                    print(
                        'Bad values found (quitting): {} \n All:{}'.format(
                            bads, routine.results))
                    exit(0)

            # Update results
            update_dict_of_lists(self._results, **routine.results)
            update_dict_of_lists(
                self._results['time'], **{key: end_time - start_time})

        update_dict_of_lists(self._results['losses'], **losses)

    def reset_routines(self):
        self._losses.clear()
        for routine in self._routines.values():
            routine.reset()

    def reset(self):
        self.reset_routines()
        self._results.clear()
        self._results.update(losses=dict(), time=dict())
        self._vars.clear()
        self._losses.clear()

    def set_train(self):
        for net in self._nets.values():
            net.train()

    def set_eval(self):
        for net in self._nets.values():
            net.eval()


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
