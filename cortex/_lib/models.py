'''Builds arch

'''

import copy
import logging
import time

from . import data, exp, optimizer
from .parsing import parse_docstring, parse_inputs, parse_kwargs
from .handlers import (aliased, prefixed, NetworkHandler, LossHandler,
                       ResultsHandler)
from .utils import bad_values, update_dict_of_lists
from .viz import VizHandler


__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.models')

MODEL_PLUGINS = {}


def register_model(plugin):
    '''

    Args:
        plugin: TODO

    Returns:
        TODO

    '''

    global MODEL_PLUGINS

    if plugin.__name__ in MODEL_PLUGINS:
        raise KeyError('{} already registered under the same name.'
                       .format(plugin.__name__))

    MODEL_PLUGINS[plugin.__name__] = plugin()


def get_model(model_name):
    try:
        return MODEL_PLUGINS[model_name]
    except KeyError:
        raise KeyError('Model {} not found. Available: {}'
                       .format(model_name, tuple(MODEL_PLUGINS.keys())))


class PluginType(type):
    def __new__(metacls, name, bases, attrs):
        cls = super(PluginType, metacls).__new__(metacls, name, bases, attrs)

        help = {}
        kwargs = {}
        args = set()

        for key in ['build', 'routine', 'visualize', 'train_step', 'eval_step']:
            if hasattr(cls, key):
                attr = getattr(cls, key)
                help_ = parse_docstring(attr)
                kwargs_ = parse_kwargs(attr)
                args_ = set(parse_inputs(attr))

                for k, v in help_.items():
                    if k in help and v != help[k]:
                        metacls._warn_inconsitent_help(key, k, v, kwargs[k])

                for k, v in kwargs_.items():
                    if k in kwargs and v != kwargs[k] and v is not None:
                        metacls._warn_inconsitent_kwargs(key, k, v, kwargs[k])

                help.update(**help_)

                for k, v in kwargs_.items():
                    if k not in kwargs or (k in kwargs and v is not None):
                        kwargs[k] = v
                args |= args_

        cls._help = help
        cls._kwargs = kwargs
        cls._args = args

        return cls

    def _warn_inconsitent_help(cls, k, v, v_):
        logger.warning('Inconsistent docstring found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))

    def _warn_inconsitent_kwargs(cls, k, v, v_):
        logger.warning('Inconsistent keyword defaults found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))


class ModelPluginBase(metaclass=PluginType):
    '''
    TODO
    '''

    _viz = VizHandler()
    _data = data.DATA_HANDLER
    _optimizers = optimizer.OPTIMIZERS

    _training_nets = dict()

    _all_nets = NetworkHandler(allow_overwrite=False)
    _all_losses = LossHandler(_all_nets, add_values=True)
    # TODO(Devon): This is done in conjunction with clearing all losses
    # after train_step.
    _all_results = ResultsHandler()

    _all_epoch_results = ResultsHandler()
    _all_epoch_losses = ResultsHandler()
    _all_epoch_times = ResultsHandler()

    def __init__(self, contract=None):
        '''

        Args:
            contract: A dictionary of strings which specify naming w.r.t.
                the model that creates this model.
        '''

        self._contract = None
        self._train = False
        self._models = []
        self.name = self.__class__.__name__

        if contract:
            contract = self._check_contract(contract)
            self._accept_contract(contract)

        if self._contract and len(self._contract['nets']) > 0:
            self._nets = aliased(self._all_nets, aliases=contract['nets'])
            self._losses = aliased(self._all_losses, aliases=contract['nets'])
            self._epoch_losses = aliased(
                self._all_epoch_losses, aliases=contract['nets'])
        else:
            self._nets = aliased(self._all_nets)
            self._losses = aliased(self._all_losses)
            self._epoch_losses = aliased(self._all_epoch_losses)

        self.wrap_functions()

        self._results = prefixed(
            self._all_results, prefix=self.name)
        self._epoch_results = prefixed(
            self._all_epoch_results, prefix=self.name)
        self._epoch_times = self._all_epoch_times

    def wrap_functions(self):
        self._wrap_routine()
        self.visualize = self._wrap(self.visualize)
        self.train_step = self._wrap_step(self.train_step)
        self.eval_step = self._wrap_step(self.eval_step, train=False)
        self.train_loop = self._wrap_loop(self.train_loop, train=True)
        self.eval_loop = self._wrap_loop(self.eval_loop, train=False)
        self.build = self._wrap(self.build)

    @classmethod
    def _reset_class(cls):
        '''Resets the static variables.

        '''
        cls._kwargs.clear()
        cls._help.clear()
        cls._training_nets.clear()

        cls._all_nets.clear()
        cls._all_losses.clear()
        cls._all_results.clear()

        cls._all_epoch_results.clear()
        cls._all_epoch_losses.clear()
        cls._all_epoch_times.clear()

    def _reset_epoch(self):
        self._all_epoch_results.clear()
        self._all_epoch_losses.clear()
        self._all_epoch_times.clear()

    def _get_id(self, fn):
        '''Gets a unique identifier for a function.

        Args:
            fn: a callable.

        Returns:
            An indetifier.

        '''
        return fn

    @property
    def kwargs(self):
        return self._kwargs

    def inputs(self, *keys):
        '''Pulls inputs from the data.

        This uses the contract to pull the right key from the data.

        Args:
            keys: List of string variable names.

        Returns:
            Tensor variables.

        '''

        if self._contract is not None:
            input_dict = self._contract['inputs']
        else:
            input_dict = {}

        inputs = []
        for k in keys:
            key = input_dict.get(k, k)
            inp = self.data[key]
            inputs.append(inp)

        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs

    @property
    def help(self):
        return self._help

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

    @property
    def data(self):
        return self._data

    def _set_train(self):
        self._train = True
        for m in self._models:
            m._set_train()

    def _set_eval(self):
        self._train = False
        for m in self._models:
            m._set_eval()

    def __setattr__(self, key, value):
        '''Sets an attribute for the model.

        Overriding is done to handle adding a ModelPlugin attribute to this
        object.

        '''
        if isinstance(value, ModelPluginBase):
            model = value
            kwargs = model.kwargs
            help = model.help

            if model._contract:
                kwargs = dict((model._contract['kwargs'].get(k, k), v)
                              for k, v in kwargs.items() if v is not None)
                help = dict((model._contract['kwargs'].get(k, k), v)
                            for k, v in help.items())

            for k, v in kwargs.items():
                if k not in self.kwargs or self.kwargs[k] is None:
                    self.kwargs[k] = copy.deepcopy(v)
            for k, v in help.items():
                if k not in self.help:
                    self.help[k] = help[k]
            model._set_kwargs(self._kwargs)
            model.name = key

            model._results = prefixed(
                model._all_results, prefix=model.name)
            model._epoch_results = prefixed(
                model._all_epoch_results, prefix=model.name)
            self._models.append(model)

        super().__setattr__(key, value)

    def _set_kwargs(self, kwargs):
        self._kwargs = kwargs
        for model in self._models:
            model._set_kwargs(kwargs)

    def _check_contract(self, contract):
        '''Checks the compatability of the contract.

        Checks the keys in the contract to make sure they correspond to inputs
        or hyperparameters of functions in this class.

        Args:
            contract: Dictionary contract.

        Returns:
            A cleaned up version of the contract.

        '''
        kwargs = contract.pop('kwargs', {})
        nets = contract.pop('nets', {})
        inputs = contract.pop('inputs', {})

        if len(contract) > 0:
            raise KeyError('Unknown keys in contract: {}'
                           .format(tuple(contract.keys())))

        for k, v in kwargs.items():
            if k not in self._kwargs:
                raise KeyError('Invalid contract: {} does not have any '
                               'arguments called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        for k, v in inputs.items():
            if k not in self._args:
                raise KeyError('Invalid contract: {} does not have any '
                               'inputs called {}'
                               .format(self.__class__.__name__, k))

            if not isinstance(v, str):
                raise TypeError('Contract values must be strings.')

        return dict(inputs=inputs, kwargs=kwargs, nets=nets)

    def _accept_contract(self, contract):
        '''Accepts the contract.

        Args:
            contract: Dictionary contract.

        '''
        if self._contract is not None:
            raise ValueError('Cannot accept more than one contract.')

        self._contract = contract

    def _wrap(self, fn):
        '''Wraps methods to allow for auto inputs and kwargs.

        Args:
            fn: A callable.

        Returns:
            A wrapped version of the callable.

        '''

        def _fetch_kwargs(**kwargs_):
            if self._contract is not None:
                kwarg_dict = self._contract['kwargs']
            else:
                kwarg_dict = {}
            kwarg_keys = parse_kwargs(fn).keys()

            kwargs = dict()
            for k in kwarg_keys:
                key = kwarg_dict.get(k, k)
                try:
                    value = self.kwargs[key]
                except KeyError:
                    value = kwargs_.get(key)
                kwargs[k] = value

            return kwargs

        def _fetch_inputs():
            if self._contract is not None:
                input_dict = self._contract['inputs']
            else:
                input_dict = {}
            input_keys = parse_inputs(fn)

            inputs = []
            for k in input_keys:
                key = input_dict.get(k, k)
                if key == 'args':
                    continue
                value = self.data[key]
                inputs.append(value)
            return inputs

        def wrapped(*args, auto_input=False, **kwargs_):
            kwargs = _fetch_kwargs(**kwargs_)
            for k, v in kwargs_.items():
                if isinstance(v, dict) and (k in kwargs and
                                            isinstance(kwargs[k], dict)):
                    kwargs[k].update(**v)
                elif v is not None:
                    kwargs[k] = v
                elif v is None and k not in kwargs:
                    kwargs[k] = v
            if auto_input:
                args = _fetch_inputs()
            return fn(*args, **kwargs)

        return wrapped

    def _wrap_routine(self):
        '''Wraps the routine.

        Set to `requires_grad` for models that are trained with this routine.

        '''

        fn = self.routine
        fn = self._wrap(fn)

        def wrapped(*args, **kwargs):
            fid = self._get_id(fn)

            if fid not in self._training_nets:
                losses_before = dict(kv for kv in self._all_losses.items())
                fn(*args, **kwargs)
                losses_after = dict(kv for kv in self._all_losses.items())

                training_nets = []

                for k, v in losses_after.items():
                    try:
                        if k not in losses_before:
                            training_nets.append(k)
                        elif v != losses_before[k]:
                            training_nets.append(k)
                    except TypeError:
                        training_nets.append(k)
                self._training_nets[fid] = training_nets
                for k in training_nets:
                    self.losses.pop(k)
            else:
                training_nets = self._training_nets[fid]

            if self._train:
                for k in training_nets:
                    net = self.nets[k]

                    optimizer = self._optimizers.get(k)
                    if optimizer is not None:
                        optimizer.zero_grad()

                    for p in net.parameters():
                        p.requires_grad = k in training_nets
                    net.train()

            start = time.time()
            output = fn(*args, **kwargs)
            self._check_bad_values()
            end = time.time()

            update_dict_of_lists(self._epoch_results, **self.results)
            update_dict_of_lists(self._epoch_times,
                                 **{self.name: end - start})
            losses = dict()
            for k, v in self.losses.items():
                if isinstance(v, (tuple, list)):
                    losses[k] = sum([v_.item() for v_ in v])
                else:
                    losses[k] = v.item()
            update_dict_of_lists(self._epoch_losses, **losses)

            return output

        self.routine = wrapped

    def _wrap_step(self, fn, train=True):
        '''Wraps the training or evaluation step.

        Args:
            fn: Callable step function.
            train (bool): For train or eval step.

        Returns:
            Wrapped version of the function.

        '''

        fn = self._wrap(fn)

        def wrapped(*args, _init=False, **kwargs):
            if train and not _init:
                self._set_train()
                for net in self.nets.values():
                    net.train()
            else:
                self._set_eval()
                for net in self.nets.values():
                    net.eval()

            output = fn(*args, **kwargs)
            self.losses.clear()

            return output

        return wrapped

    def _wrap_loop(self, fn, train=True):
        '''Wraps a loop.

        Args:
            fn: Callable loop function.
            train (bool): For train or eval loop.

        Returns:
            Wrapped version of the function.

        '''

        data_mode = 'train' if train else 'test'

        if train:
            epoch_str = 'Training {} (epoch {}): '
        else:
            epoch_str = 'Evaluating {} (epoch {}): '

        def wrapped(epoch, data_mode=data_mode, use_pbar=True):
            self._reset_epoch()
            self.data.reset(data_mode, string=epoch_str.format(exp.NAME, epoch),
                            make_pbar=use_pbar)
            fn()

            results = self._all_epoch_results
            results['losses'] = dict(self._all_epoch_losses)
            results['times'] = dict(self._all_epoch_times)

        return wrapped

    def _get_training_nets(self):
        '''Retrieves the training nets for the object.

        '''

        training_nets = []
        for v in self._training_nets.values():
            training_nets += v

        return training_nets

    def _check_bad_values(self):
        '''Check for bad numbers.

        This checks the results and the losses for nan or inf.

        '''

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

    def reload_nets(self, nets_to_reload):
        if nets_to_reload:
            self.nets._handler.load(**nets_to_reload)
