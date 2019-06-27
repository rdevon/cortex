'''Builds arch

'''

import copy
import logging
import time

import torch

from . import data, exp, optimizer, viz
from .parsing import parse_docstring, parse_inputs, parse_kwargs
from .handlers import nested, NetworkHandler, LossHandler
from .utils import bad_values, update_dict_of_lists


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
        '''

        This effectively reads all the keyword arguments and docstring from core functions,
        then collects these and makes sure they are consistent for a model.

        '''
        cls = super(PluginType, metacls).__new__(metacls, name, bases, attrs)

        help = {}
        kwargs = {}
        args = set()

        # TODO change this to test for decorator, as per future changes to model.
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

        # Get parent class arguemnts, kwargs, and help. TODO
        '''
        parents = cls.__bases__
        for parent in parents:
            if hasattr(parent, '_kwargs'):
                for k, v in parent._kwargs.items():
                    if k in kwargs and v != kwargs[k] and v is not None:
                        metacls._warn_inconsitent_kwargs(key, k, kwargs[k], v)
                    else:
                        kwargs[k] = v
            if hasattr(parent, '_help'):
                for k, v in parent._help.items():
                    if k in help and v != help[k]:
                        metacls._warn_inconsitent_help(key, k, kwargs[k], v)
                    else:
                        help[k] = v
        '''

        cls._help = help
        cls._kwargs = kwargs
        cls._args = args

        return cls

    def _warn_inconsitent_help(cls, k, v, v_):
        logger.warning('Inconsistent docstring found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))

    def _warn_inconsitent_kwargs(cls, k, v, v_):
        logger.warning('Inconsistent hyperparameter defaults found with argument {k}. '
                       'Using {v} instead of {v_}'.format(k=k, v=v, v_=v_))


def _auto_input_decorator(cls, fn):
    '''Wraps methods to allow for auto inputs and kwargs.

    Args:
        fn: A callable.

    Returns:
        A wrapped version of the callable.

    '''

    if not callable(fn):
        raise ValueError('Only callables (functions) can be wrapped.')

    def _fetch_kwargs(**kwargs_):
        if cls._contract is not None:
            kwarg_dict = cls._contract['kwargs']
        else:
            kwarg_dict = {}
        kwarg_keys = parse_kwargs(fn).keys()

        kwargs = dict()
        for k in kwarg_keys:
            key = kwarg_dict.get(k, k)
            try:
                value = cls._hyperparams[key]
            except KeyError:
                value = kwargs_.get(key)
            kwargs[k] = value

        return kwargs

    def _fetch_inputs():
        if cls._contract is not None:
            input_dict = cls._contract['inputs']
        else:
            input_dict = {}
        input_keys = parse_inputs(fn)

        inputs = []
        for k in input_keys:
            key = input_dict.get(k, k)
            if key == 'args':
                continue
            value = cls.data[key]
            inputs.append(value)
        return inputs

    def wrapped(*args, auto_input=False, **kwargs_):
        kwargs = _fetch_kwargs(**kwargs_)
        for k, v in kwargs_.items():
            # Infer correct hyperparameters for function and pass them..
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


def model_step(cls, fn, train=True):
    '''Wraps the training or evaluation step.

    Args:
        fn: Callable step function.
        train (bool): For train or eval step.

    Returns:
        Wrapped version of the function.

    '''

    fn = _auto_input_decorator(cls, fn)

    def cortex_step(*args, _init=False, **kwargs):
        if train and not _init:
            training_nets = cls._get_training_nets()
            for k, net in cls._all_nets.items():
                if k in training_nets:
                    net.train()
                else:
                    net.eval()
        else:
            for net in cls._all_nets.values():
                net.eval()

        if train:
            output = fn(*args, **kwargs)
            cls.finish_step()
        else:
            with torch.no_grad():
                output = fn(*args, **kwargs)
                cls.finish_step()

        return output

    return cortex_step


def cortex_train_step(fn):
    def cortex_step(cls, *args, _init=False, **kwargs):
        if not _init:
            training_nets = cls._get_training_nets()
            for k, net in cls._all_nets.items():
                if k in training_nets:
                    net.train()
                else:
                    net.eval()
        else:
            for net in cls._all_nets.values():
                net.eval()

        output = _auto_input_decorator(cls, fn)(cls, *args, **kwargs)
        cls.finish_step()

        return output

    return cortex_step


def cortex_eval_step(fn):
    def cortex_step(cls, *args, _init=False, **kwargs):
        for net in cls._all_nets.values():
            net.eval()

        with torch.no_grad():
            output = _auto_input_decorator(cls, fn)(cls, *args, **kwargs)
            cls.finish_step()

        return output

    return cortex_step


class ModelPluginBase(metaclass=PluginType):
    '''Base model class.

    '''

    # Global attributes that all models have access to.
    _data = data.DATA_HANDLER
    _optimizers = optimizer.OPTIMIZERS
    _all_nets = NetworkHandler()
    _viz = viz.viz_handler

    _steps = ['train_step']

    def __init__(self, kwargs=None, nets=None, inputs=None):
        '''

        Args:
            kwargs: mapping for hyperparameter names.
            nets: name mapping for networks.
            inputs: name mapping for input names.
        '''

        self._contract = None
        self._models = []
        self.name = self.__class__.__name__

        self._hyperparams = copy.deepcopy(self._kwargs)
        self._training_nets = None

        # Contract to handle naming of things.
        kwargs = kwargs or {}
        nets = nets or {}
        inputs = inputs or {}
        contract = dict(inputs=inputs, nets=nets, kwargs=kwargs)
        self._contract = contract

        # Network handler.
        self._nets = nested(self._all_nets, self)  # Networks.

        # Handlers for various results.
        self._results = dict()  # Results to be displayed.
        self._grads = dict() # Gradients of each optimizer.
        self._times = dict()  # Times for models.
        self._losses = LossHandler(self.nets)  # Losses for training.
        self._all_losses = dict()  # Summary of losses for display.

        # Wrap functions
        self.build = _auto_input_decorator(self, self.build)
        self.visualize = self.visualize_step(self.visualize)
        self.routine = self.model_routine(self.routine)
        self.optimizer_step = self.model_optimizer_step(self.optimizer_step)
        self.train_step = model_step(self, self.train_step, train=True)
        self.eval_step = model_step(self, self.eval_step, train=False)
        self.train_loop = self.model_loop(self.train_loop, train=True)
        self.eval_loop = self.model_loop(self.eval_loop, train=False)

    def __setattr__(self, key, value):
        '''Sets an attribute for the model.

        Overriding is done to handle adding a ModelPlugin attribute to this
        object.

        '''
        if isinstance(value, ModelPluginBase):
            model = value
            model.name = key
            self._models.append(model)

        super().__setattr__(key, value)

    def add_models(self, **kwargs):
        for k, v in kwargs.items():
            self.add_model(k, v)

    def add_model(self, key, value):
        if isinstance(value, ModelPluginBase):
            setattr(self, key, value)
        else:
            raise TypeError('Model must be subclass of {}'.format(ModelPluginBase))

    # Add results and losses
    def add_results(self, **kwargs):
        for key, value in kwargs.items():
            self.results[key] = value

    def add_losses(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.losses.keys():
                v = self.losses[key]
                self.losses[key] = [v, value]
            else:
                self.losses[key] = value

    def add_nets(self, **kwargs):
        for key, value in kwargs.items():
            self.nets[key] = value

    def add_grads(self, **kwargs):
        for key, value in kwargs.items():
            self.grads[key] = value

    # Hyperparameter functions
    def pull_hyperparameters(self):
        hyperparameters = copy.deepcopy(self._hyperparams)
        for model in self._models:
            m_hypers = model.pull_hyperparameters()
            hyperparameters[model.name] = m_hypers

        return hyperparameters

    def push_hyperparameters(self, hyperameters):
        for k, v in hyperameters.items():
            if k in self._hyperparams.keys():
                self._hyperparams[k] = v

        for child in self._models:
            child_hypers = hyperameters.get(child.name, {})
            child.push_hyperparameters(child_hypers)

    def pull_info(self):
        # TODO
        info = copy.deepcopy(self._help)
        for model in self._models:
            m_info = model.pull_info()
            info[model.name] = m_info

        return info

    # Functions for pulling and collapsing results

    def pull_losses(self):
        '''Get all network losses from model and child models.

        Returns:
            dict: Dictionary of network losses and children losses.

        '''
        losses = dict((self.nets._aliases[k], v)
                      for k, v in self.losses.items())
        self.losses.clear()
        for model in self._models:
            losses[model.name] = model.pull_losses()
        return losses

    def pull_results(self):
        '''Get all results from model and child models.

        Returns:
            dict: Dictionary of results and children results.

        '''
        results = dict((self._contract['nets'].get(k, k), v)
                       for k, v in self.results.items())
        self.results.clear()
        for model in self._models:

            results[model.name] = model.pull_results()
        return results

    def pull_grads(self):
        '''Get all grads from model and child models.

        Returns:
            dict: Dictionary of grads and children grads.

        '''
        grads = dict((self._contract['nets'].get(k, k), v)
                       for k, v in self.grads.items())
        self.grads.clear()
        for model in self._models:
            grads[model.name] = model.pull_grads()
        return grads

    def pull_times(self):
        '''Get all times from model and child models.

        Returns:
            dict: Dictionary of routine times and children times.

        '''
        times = dict((self._contract['nets'].get(k, k), v)
                     for k, v in self.times.items())
        self.times.clear()
        for model in self._models:
            times[model.name] = model.pull_times()
        return times

    # Routine, step, and loop wrappers
    def model_routine(self, fn):
        '''Wraps the routine.

        '''

        fn = _auto_input_decorator(self, fn)

        def cortex_routine(*args, **kwargs):
            start = time.time()
            output = fn(*args, **kwargs)
            if self._training_nets is None:
                self._training_nets = [self.nets._aliases.get(k, k) for k in self.losses.keys()]
            self._check_bad_values()
            end = time.time()
            if self.name in self.times:
                self.times[self.name] += (end - start)
            else:
                self.times[self.name] = end - start

            return output

        return cortex_routine

    def visualize_step(self, fn):

        fn = _auto_input_decorator(self, fn)

        def viz_step(*args, **kwargs):

            for k, net in self._all_nets.items():
                net.eval()
            output = fn(*args, **kwargs)
            self.clear()
            return output

        return viz_step

    def finish_step(self):
        '''Finishes a step.

        Should only be done when all models have completed their step.

        '''
        # Add results to the global results.
        times = self.pull_times()
        grads = self.pull_grads()
        results = self.pull_results()
        losses = self.pull_losses()

        exp.RESULTS.update(mode=self.data.mode,
                           results=results,
                           times=times,
                           grads=grads)

        exp.RESULTS.update_losses(losses, mode=self.data.mode)
        self.viz.update(self.visualize)

        self.clear()

    def model_loop(self, fn, train=True):
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

        def cortex_loop(epoch, data_mode=data_mode, use_pbar=True):
            self.data.reset(data_mode, string=epoch_str.format(exp.NAME, epoch),
                            make_pbar=use_pbar)

            if data_mode == 'train':
                # Normal learning rate scheduler.
                for k, sched in optimizer.SCHEDULERS.items():
                    lr = optimizer.OPTIMIZERS[k].param_groups[0]['lr']
                    if not isinstance(
                            sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        sched.step()
                    lr_ = optimizer.OPTIMIZERS[k].param_groups[0]['lr']
                    if lr != lr_:
                        logger.debug(
                            'Learning rate for {} changed to {}'.format(k, lr_))

            fn()

            # Optimizer scheduler.
            if data_mode == 'train':
                # Plateau LR based scheduler.
                for k, sched in optimizer.SCHEDULERS.items():
                    lr = optimizer.OPTIMIZERS[k].param_groups[0]['lr']
                    if isinstance(
                            sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        loss = exp.RESULTS.losses[k][-1]
                        sched.step(loss)
                    lr_ = optimizer.OPTIMIZERS[k].param_groups[0]['lr']
                    if lr != lr_:
                        logger.debug(
                            'Learning rate for {} changed to {}'.format(k, lr_))

        return cortex_loop

    def model_optimizer_step(self, fn):
        '''Collapses losses from children models and the runs optimization.

        Args:
            fn: function to wrap.

        '''

        def cortex_optimizer_step(*args, **kwargs):

            def collapse(d, losses):
                for k, v in d.items():
                    if isinstance(v, dict):
                        collapse(v, losses)
                    else:
                        if k in losses.keys():
                            losses[k] += v
                        else:
                            losses[k] = v

            losses = self.pull_losses()  # Pull losses from models.

            # Push losses to the results.
            exp.RESULTS.update_losses(losses, mode=self.data.mode)

            # Collect losses by network and add.
            losses_by_network = {}
            collapse(losses, losses_by_network)
            self.losses.update(**losses_by_network)

            fn(*args, **kwargs)  # Optimizer step.
            self.losses.clear()  # Cleanup

        return cortex_optimizer_step

    def _map_data_queries(self, *queries):
        """Maps a query for an input for a model to a different key.

        Args:
            *queries: list of keys.

        Returns:
            New list of keys.

        """

        queries_ = []
        for query in queries:
            query = self._contract['inputs'].get(query, query)
            queries_.append(query)

        return queries_

    def _get_training_nets(self):
        '''Retrieves the training nets for the object.

        '''

        if self._training_nets is not None:
            training_nets = self._training_nets[:]
        else:
            training_nets = []

        for child in self._models:
            training_nets += child._get_training_nets()

        return training_nets

    def inputs(self, *keys):
        '''Pulls inputs from the data.

        This uses the contract to pull the right key from the data.

        Args:
            keys: List of string variable names.

        Returns:
            Tensor variables.

        '''

        keys = self._map_data_queries(*keys)

        inputs = []
        for k in keys:
            inp = self.data[k]
            inputs.append(inp)

        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs

    def uniterated_inputs(self, *keys, mode=None, idx=None, idx_mode='random', N=None):
        '''Pulls test inputs from the data.

        This function exists to allow pulling test examples for visualization.
        TODO (devon): iterators need to be a little more transportable, such that the user can pull whatever data they'd like.

        Args:
            keys: List of string variable names.
            mode: If set, chose a data point from this mode.
            idx: If set, pull specific data index.
            idx_mode: How to pull idxs if they aren't provided. Supported are `random` and `firstN`
            N: number of samples, if idx is not set. If None, use batch_size

        Returns:
            Tensor variables.

        '''

        mode = mode or self.data.mode

        keys = self._map_data_queries(*keys)

        inputs = []
        for k in keys:
            # Get input and idx if it wasn't specified.
            inp, idx = self.data.get_uniterated_data(k, mode=mode, idx=idx, idx_mode=idx_mode, N=N)
            inputs.append(inp)

        if len(inputs) == 0:
            return None
        elif len(inputs) == 1:
            return inputs[0]
        else:
            return inputs

    # Properties
    @property
    def kwargs(self):
        return self._kwargs

    @property
    def help(self):
        return self._help

    @property
    def results(self):
        return self._results

    @property
    def grads(self):
        return self._grads

    @property
    def nets(self):
        return self._nets

    @property
    def losses(self):
        return self._losses

    @property
    def times(self):
        return self._times

    @property
    def viz(self):
        return self._viz

    @property
    def data(self):
        return self._data

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

    def reload_nets(self, nets_to_reload, lax_reload=False):
        if nets_to_reload:
            self.nets._handler.load(lax_reload, **nets_to_reload)

    def clear(self):
        self.times.clear()
        self.losses.clear()
        self.results.clear()
        self.grads.clear()

    def clear_viz(self):
        self._viz.clear()
