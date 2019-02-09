'''Builds arch

'''

import copy
import logging
import time

import torch

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
                value = cls.kwargs[key]
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


class ModelPluginBase(metaclass=PluginType):
    '''Base model class.

    '''

    # Global attributes that all models have access to.
    _viz = VizHandler()
    _data = data.DATA_HANDLER
    _optimizers = optimizer.OPTIMIZERS
    _hyperparams = dict()
    _training_nets = dict()
    _all_nets = NetworkHandler(allow_overwrite=False)

    # Results for an epoch.
    _epoch_losses = ResultsHandler()
    _epoch_times = ResultsHandler()
    _epoch_results = ResultsHandler()

    def __init__(self, kwargs=None, nets=None, inputs=None):
        '''

        Args:
            kwargs: mapping for hyperparameter names.
            nets: name mapping for networks.
            inputs: name mapping for input names.
        '''

        self._contract = None
        self._train = False
        self._models = []
        self.name = self.__class__.__name__

        # Contract to handle naming of things.
        kwargs = kwargs or {}
        nets = nets or {}
        inputs = inputs or {}
        contract = dict(inputs=inputs, nets=nets, kwargs=kwargs)
        self._contract = contract

        if self._contract and len(self._contract['nets']) > 0:
            self._nets = aliased(self._all_nets, aliases=contract['nets'])
        else:
            self._nets = aliased(self._all_nets)

        # Handlers for various results.
        self._results = ResultsHandler()
        self._times = ResultsHandler()
        self._losses = LossHandler(self.nets)

        # Wrap functions
        self.build = _auto_input_decorator(self, self.build)
        self.visualize = _auto_input_decorator(self, self.visualize)

        self.routine = self.model_routine(self.routine)
        self.train_step = self.model_step(self.train_step, train=True)
        self.optimizer_step = self.model_optimizer_step(self.optimizer_step)
        self.eval_step = self.model_step(self.eval_step, train=False)
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

    # Hyperparameter functions
    def pull_hyperparameters(self):
        hyperparameters = copy.deepcopy(self._kwargs)
        for model in self._models:
            m_hypers = model.pull_hyperparameters()
            m_hypers = dict(('{}.{}'.format(model.name, k), v)
                             for k, v in m_hypers.items())
            hyperparameters.update(**m_hypers)

        return hyperparameters

    def push_hyperparameters(self, hyperameters):
        for k, v in hyperameters.items():
            if k in self.kwargs.keys():
                self.kwargs[k] = v

        for child in self._models:
            child_hypers = hyperameters.get(child.name, {})
            child.push_hyperparameters(child_hypers)

    def pull_info(self):
        # TODO
        info = copy.deepcopy(self._help)
        for model in self._models:
            m_info = model.pull_info()
            m_info = dict(('{}.{}'.format(model.name, k), v)
                          for k, v in m_info.items())
            info.update(**m_info)

        return info

    # Functions for fetching and collapsing results

    def fetch_all_losses(self):
        '''Get all network losses from model and child models.

        Returns:
            dict: Dictionary of network losses and children losses.

        '''
        losses = dict((self._contract['nets'].get(k, k), v)
                      for k, v in self.losses.items())
        self.losses.clear()
        for model in self._models:
            losses[model.name] = model.fetch_all_losses()
        return losses

    def fetch_all_results(self):
        '''Get all results from model and child models.

        Returns:
            dict: Dictionary of results and children results.

        '''
        results = dict((self._contract['nets'].get(k, k), v)
                       for k, v in self.results.items())
        self.results.clear()
        for model in self._models:
            results[model.name] = model.fetch_all_results()
        return results

    def fetch_all_times(self):
        '''Get all times from model and child models.

        Returns:
            dict: Dictionary of routine times and children times.

        '''
        times = dict((self._contract['nets'].get(k, k), v)
                     for k, v in self.times.items())
        self.times.clear()
        for model in self._models:
            times[model.name] = model.fetch_all_times()
        return times

    def collapse_losses(self):
        '''Collapses network losses from child models.

        '''
        def collapse(d, losses):
            for k, v in d.items():
                if isinstance(v, dict):
                    collapse(v, losses)
                elif k in losses.keys():
                    losses[k] += v
                else:
                    losses[k] = v

        losses = self.fetch_all_losses()
        losses_ = {}
        collapse(losses, losses_)
        self.losses.clear()
        self.losses.update(**losses_)

    def collapse_results(self):
        '''Collapses results from child models.

        '''
        def collapse(d, results, prefix=''):
            for k, v in d.items():
                if prefix == '':
                    key = k
                else:
                    key = prefix + '.' + k
                if isinstance(v, dict):
                    collapse(v, results, prefix=key)
                else:
                    results[key] = v
        results = self.fetch_all_results()
        results_ = {}
        collapse(results, results_)
        self._results.clear()
        self._results.update(**results_)

    def collapse_times(self):
        '''Collapses times from child models.

        '''
        def collapse(d, times, key=None):
            for k, v in d.items():
                key_ = key or k
                if isinstance(v, dict):
                    collapse(v, times, key_)
                else:
                    times[key_] = v

        times = self.fetch_all_times()
        times_ = {}
        collapse(times, times_)
        self._times = times_

    # Routine, step, and loop wrappers
    def model_routine(self, fn):
        '''Wraps the routine.

        Set to `requires_grad` for models that are trained with this routine.

        '''

        fn = _auto_input_decorator(self, fn)

        def cortex_routine(*args, **kwargs):
            fid = self._get_id(fn)

            if fid not in self._training_nets:
                # This is the first time a routine has been run. Run once,
                # then extract the networks from which losses are computed for.
                # These will be trained.
                with torch.cuda.device(exp.DEVICE.index):
                    fn(*args, **kwargs)
                training_nets = [self._contract['nets'].get(k, k) for k in self.losses.keys()]
                self._training_nets[fid] = training_nets
            else:
                training_nets = self._training_nets[fid]

            self.losses.clear()

            if self._train:
                for k in training_nets:
                    net = self.nets[k]

                    optimizer = self._optimizers.get(k)
                    if optimizer is not None:
                        optimizer.zero_grad()

                    for p in net.parameters():
                        p.requires_grad = True
                    net.train()

            start = time.time()
            with torch.cuda.device(exp.DEVICE.index):
                output = fn(*args, **kwargs)

            self._check_bad_values()
            end = time.time()
            self.times[self.__class__.__name__] = end - start

            return output

        return cortex_routine

    def model_step(self, fn, train=True):
        '''Wraps the training or evaluation step.

        Args:
            fn: Callable step function.
            train (bool): For train or eval step.

        Returns:
            Wrapped version of the function.

        '''

        fn = _auto_input_decorator(self, fn)

        def cortex_step(*args, _init=False, **kwargs):
            if train and not _init:
                self._set_train()
                for net in self.nets.values():
                    net.train()
            else:
                self._set_eval()
                for net in self.nets.values():
                    net.eval()

            output = fn(*args, **kwargs)

            # Collapse results and times
            self.collapse_results()
            self.collapse_losses()
            self.collapse_times()
            return output

        return cortex_step

    def finish_step(self):
        '''Finishes a loop.

        Should only be done when all models have completed their step.

        '''
        # Appends to each list of results and losses.
        loss_dict = dict((k, v.detach().item()) for k, v in self.losses.items())

        update_dict_of_lists(self._epoch_results, **self.results)
        update_dict_of_lists(self._epoch_times, **self.times)
        update_dict_of_lists(self._epoch_losses, **loss_dict)

        self.losses.clear()
        self.results.clear()
        self.times.clear()

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
            self._reset_epoch()
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

            results = self._epoch_results
            results['losses'] = dict(self._epoch_losses)
            results['times'] = dict(self._epoch_times)

            # Optimizer scheduler.
            if data_mode == 'train':
                # Plateau LR based scheduler.
                for k, sched in optimizer.SCHEDULERS.items():
                    lr = optimizer.OPTIMIZERS[k].param_groups[0]['lr']
                    if isinstance(
                            sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        loss = results['losses'][k][-1]
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

        def cortex_optimizer_step():
            self.collapse_losses()
            start = time.time()
            fn()
            end = time.time()
            self.times['Optimizer'] = end - start

        return cortex_optimizer_step

    # Resetting
    @classmethod
    def _reset_class(cls):
        '''Resets the static variables.

        '''
        cls._kwargs.clear()
        cls._help.clear()
        cls._training_nets.clear()
        cls._all_nets.clear()

        cls._epoch_results.clear()
        cls._epoch_losses.clear()
        cls._epoch_times.clear()

    def _reset_epoch(self):
        self._epoch_results.clear()
        self._epoch_losses.clear()
        self._epoch_times.clear()

    # Get info and other queries
    def _get_id(self, fn):
        '''Gets a unique identifier for a function.

        Args:
            fn: a callable.

        Returns:
            An indentifier.

        '''
        return fn

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

        training_nets = []
        for v in self._training_nets.values():
            training_nets += v

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

    def _set_train(self):
        self._train = True
        for m in self._models:
            m._set_train()

    def _set_eval(self):
        self._train = False
        for m in self._models:
            m._set_eval()

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

    def clear_viz(self):
        self._viz.clear()
