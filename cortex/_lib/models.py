'''Builds arch

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import importlib
import logging
import os
import sys
import time

import torch

from . import data, exp, models, optimizer
from .parsing import parse_docstring, parse_header, parse_kwargs
from .handlers import Handler, NetworkHandler, LossHandler, ResultsHandler
from .utils import bad_values, update_dict_of_lists
from .viz import VizHandler


logger = logging.getLogger('cortex.models')
MODEL = None

_ROUTINE_PLUGINS = {}
_BUILD_PLUGINS = {}
MODEL_PLUGINS = {}
NETWORK_HANDLER = NetworkHandler()


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


def register_build(plugin):
    global _BUILD_PLUGINS
    check_plugin(plugin, 'build', _BUILD_PLUGINS)

    plugin.help = parse_docstring(plugin.build)
    plugin.kwargs = parse_kwargs(plugin.build)

    _BUILD_PLUGINS[plugin.plugin_name] = plugin


def register_routine(plugin):
    global _ROUTINE_PLUGINS
    check_plugin(plugin, 'routine', _ROUTINE_PLUGINS)

    plugin.help = parse_docstring(plugin.run)
    plugin.kwargs = parse_kwargs(plugin.run)

    _ROUTINE_PLUGINS[plugin.plugin_name] = plugin


def register_model(plugin):
    global MODEL_PLUGINS
    check_plugin(plugin, 'model', MODEL_PLUGINS)

    plugin.help, plugin.description = parse_header(plugin)

    MODEL_PLUGINS[plugin.plugin_name] = plugin


class BuildReference():
    def __init__(self, key, **kwargs):
        self.reference = key
        self.kwargs = kwargs

    def resolve(self):
        try:
            plugin = _BUILD_PLUGINS[self.reference]
        except KeyError:
            raise KeyError(
                'Build `{}` not registered in cortex. '
                'Available: {}'.format(
                    self.reference, tuple(
                        _BUILD_PLUGINS.keys())))
        return plugin(**self.kwargs)


class RoutineReference():
    def __init__(self, key, **kwargs):
        self.reference = key
        self.kwargs = kwargs

    def resolve(self):
        try:
            plugin = _ROUTINE_PLUGINS[self.reference]
        except KeyError:
            raise KeyError(
                'Routine `{}` not registered in cortex. '
                'Available: {}'.format(
                    self.reference, tuple(
                        _ROUTINE_PLUGINS.keys())))
        return plugin(**self.kwargs)


class BuildPluginBase():
    def __init__(self, **kwargs):
        self._data = data.DATA_HANDLER
        self._nets = models.NETWORK_HANDLER
        self._names = {}

        keys = self.plugin_nets
        for k, v in kwargs.items():
            # TODO(Devon): might have to do checking for keys here.
            if k in self._names:
                raise KeyError('`{}` is already set'.format(k))
            self._names[k] = v

        kwargs_ = {}
        for k, v in self.kwargs.items():
            k_ = self._names.get(k, k)
            kwargs_[k_] = v
        self.kwargs = kwargs_

    def __call__(self, **kwargs):
        if not hasattr(self, 'build'):
            raise ValueError(
                'Build {} does not have `build` method set'.format(
                    self.name))
        kwargs_ = {}
        names = dict((v, k) for k, v in self._names.items())
        for k, v in kwargs.items():
            k_ = names.get(k, k)
            kwargs_[k_] = v
        self.build(**kwargs_)


class RoutinePluginBase():
    _training_models = []

    def __init__(self, name=None, **kwargs):
        self._names = {}
        self.nets = NetworkHandler()
        self.results = ResultsHandler()
        self.losses = LossHandler(self.nets)
        self.inputs = Handler()
        self.training_nets = []
        self.name = name or self.plugin_name
        self.viz = None

        keys = self.plugin_nets + self.plugin_inputs + \
            self.plugin_outputs + self.plugin_optional_inputs
        for k, v in kwargs.items():
            if k not in keys:
                raise KeyError(
                    '`{}` not supported for this plugin. Available: {}'.format(
                        k, keys))
            if k in self._names:
                raise KeyError('`{}` is already set'.format(k))
            self._names[k] = v

    def __call__(self, **kwargs):
        if not hasattr(self, 'run'):
            raise ValueError(
                'Routine {} does not have `run` method set'.format(
                    self.name))
        outputs = self.run(**kwargs)
        if outputs is None:
            return {}
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.plugin_outputs):
            raise ValueError(
                'Routine has different number of outputs ({}) '
                'than is set from `plugin_outputs` ({}).'.format(
                    len(outputs), len(
                        self.plugin_outputs)))

        out_dict = {}
        for k, v in zip(self.plugin_outputs, outputs):
            k_ = self._names.get(k, k)
            out_dict[k_] = v
        return out_dict

    def perform_routine(self, **kwargs):
        # Run routine
        if exp.DEVICE == torch.device('cpu'):
            return self(**kwargs)
        else:
            with torch.cuda.device(exp.DEVICE.index):
                return self(**kwargs)

    def reset(self):
        self.results.clear()
        self.losses.clear()
        self.inputs.clear()

    def set_viz(self, viz):
        self._viz = viz


class ModelPluginBase():
    def __init__(self):
        self._builds = Handler()
        self._routines = Handler()
        self._defaults = dict(
            data=self.data_defaults,
            optimizer=self.optimizer_defaults,
            train=self.train_defaults)
        self._train_procedures = []
        self._eval_procedures = []

        self._setup = None

        self._results = ResultsHandler(time=dict(), losses=dict())
        self._nets = models.NETWORK_HANDLER
        self._losses = LossHandler(self._nets)
        self._data = data.DATA_HANDLER
        self._kwargs = {}

    @property
    def defaults(self):
        return self._defaults

    @property
    def kwargs(self):
        return self._kwargs

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

    def check(self):
        for key in self._routines.keys():
            routine = self._routines[key]
            if isinstance(routine, RoutinePluginBase):
                pass
            elif isinstance(routine, RoutineReference):
                self._routines[key] = routine.resolve()
            else:
                raise ValueError

        for key in self._builds.keys():
            build = self._builds[key]
            if isinstance(build, BuildPluginBase):
                pass
            elif isinstance(build, BuildReference):
                self._builds[key] = build.resolve()
            else:
                raise ValueError

    def get_kwargs(self):
        kwargs = {}

        def add_kwargs(obj):
            for k, v in obj.kwargs.items():
                k_ = obj._names.get(k, k)
                if k_ in kwargs:
                    if v is None:
                        pass
                    elif kwargs[k_] is None:
                        kwargs[k_] = v
                    elif kwargs[k_] != v:
                        logger.warning('Multiple default values found for {}. '
                                       'This may have unintended effects. '
                                       'Using {}'.format(k_, kwargs[k_]))
                else:
                    kwargs[k_] = v

        for build in self._builds.values():
            add_kwargs(build)

        for routine in self._routines.values():
            add_kwargs(routine)
        return kwargs

    def get_help(self):
        helps = {}

        def add_help(obj):
            for k, v in obj.help.items():
                k_ = obj._names.get(k, k)
                if k_ in helps:
                    if v is None:
                        pass
                    elif helps[k_] is None:
                        helps[k_] = v
                    elif helps[k_] != v:
                        logger.warning('Multiple '
                                       'default values found'
                                       'for {} help.'
                                       'This may have'
                                       'unintended'
                                       'effects. Using {}'
                                       .format(k_, helps[k_]))
                else:
                    helps[k_] = v

        for build in self.builds.values():
            add_help(build)

        for routine in self.routines.values():
            add_help(routine)

        return helps

    def unpack_args(self):
        builds = Handler()
        routines = Handler()

        kwargs = self.get_kwargs()

        for key, build in self.builds.items():
            for k_, v in kwargs.items():
                if k_ in build.kwargs:
                    if key in builds:
                        builds[key][k_] = v
                    else:
                        builds[key] = {k_: v}

        for key, routine in self._routines.items():
            for k_, v in kwargs.items():
                if k_ in routine.kwargs:
                    if key in routines:
                        routines[key][k_] = v
                    else:
                        routines[key] = {k_: v}

        return Handler(builds=builds, routines=routines)

    def train(self, i, quit_on_bad_values=False):
        return self.run_procedure(
            i, quit_on_bad_values=quit_on_bad_values, train=True)

    def run_procedure(self, i, quit_on_bad_values=False, train=False):
        self._data.next()
        self.reset_routines()
        inputs = Handler()
        mode, procedure, updates = self._train_procedures[i]

        for k, v in self._data.batch.items():
            inputs['data.' + k] = v

        for key, update in zip(procedure, updates):
            if not train:
                update = 1
            for u in range(update):
                if u > 0:
                    self._data.next()

                routine = self._routines[key]
                kwargs = self._kwargs[key]
                routine.reset()

                # Set to `requires_grad` for models that are trained with this
                # routine.
                if train:
                    for k in routine.training_nets:
                        k_ = routine._names.get(k, k)
                        optimizer.OPTIMIZERS[k_].zero_grad()
                        net = routine.nets[k]
                        for p in net.parameters():
                            p.requires_grad = True

                # Required inputs
                receives = routine.plugin_inputs
                sends = [routine._names.get(k, k) for k in receives]
                for send, receive in zip(sends, receives):
                    try:
                        if isinstance(send, (list, tuple)):
                            send_ = [inputs[s] for s in send]
                            routine.inputs[receive] = send_
                        else:
                            routine.inputs[receive] = inputs[send]
                    except KeyError:
                        raise KeyError(
                            '{} not found in inputs. Available: {}'.format(
                                send, tuple(
                                    inputs.keys())))

                # Optional inputs
                receives = routine.plugin_optional_inputs
                sends = [routine._names.get(k, k) for k in receives]
                for send, receive in zip(sends, receives):
                    try:
                        if isinstance(send, (list, tuple)):
                            send_ = [inputs[s] for s in send]
                            routine.inputs[receive] = send_
                        else:
                            routine.inputs[receive] = inputs[send]
                    except BaseException:
                        routine.inputs[receive] = None

                start_time = time.time()
                outputs = routine.perform_routine(**kwargs)

                # Backprop the losses.
                if train:
                    for k, loss in routine.losses.items():
                        if loss is not None:
                            loss.backward()
                            k_ = routine._names.get(k, k)
                            optimizer.OPTIMIZERS[k_].step()

                end_time = time.time()

                # Populate the inputs with the outputs
                if u == update - 1:
                    for k, v in outputs.items():
                        k_ = key + '.' + k
                        if k_ in inputs:
                            raise KeyError('{} already in'
                                           ' inputs. Use a '
                                           'different name.'.format(k_))
                        inputs[k_] = v.detach()

                # Add losses to the results.
                for loss_key in routine.losses.keys():
                    if loss_key not in routine.training_nets:
                        routine.training_nets.append(loss_key)

                # Check for bad numbers
                bads = bad_values(routine.results)
                if bads and quit_on_bad_values:
                    print(
                        'Bad values found (quitting): {} \n All:{}'.format(
                            bads, routine.results))
                    exit(0)

            routine_losses = dict((k, v.item())
                                  for k, v in routine.losses.items())

            # Update results
            update_dict_of_lists(self._results, **routine.results)
            update_dict_of_lists(self._results['losses'], **routine_losses)
            update_dict_of_lists(
                self._results['time'], **{key: end_time - start_time})

    def setup_routine_nets(self):
        self.viz = VizHandler()
        for routine in self._routines.values():
            for k in routine.plugin_nets:
                k_ = routine._names.get(k, k)
                routine.nets[k] = self._nets[k_]
            routine.set_viz(self.viz)

    def reset_routines(self):
        self._losses.clear()
        for routine in self._routines.values():
            routine.reset()

    def reset(self):
        self.reset_routines()
        self._results.clear()
        self._results.update(losses=dict(), time=dict())

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


def build_networks(**build_args):
    '''Builds the generator and discriminator.

    If architecture module contains a `build_model` function, use that,
    otherwise, use the one found in this module.

    '''
    for build_key, build in MODEL.builds.items():
        args = build_args[build_key]
        logger.debug('{} build args: {}'.format(build_key, args))
        build(**args)

    MODEL.setup_routine_nets()


def reload_models(**reload_models):
    global MODEL_HANDLER
    if MODEL_HANDLER is None:
        raise RuntimeError(
            'MODEL_HANDLER not set. `reload_models` should only be used after '
            '`models.setup_models` has been called.')
    for k, v in reload_models.items():
        logger.info('Reloading model {}'.format(k))
        logger.debug(v)
        MODEL_HANDLER[k] = v
