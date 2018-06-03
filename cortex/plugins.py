'''Module for plugins

'''

import inspect
from os import path
import shutil
import time

import torch

from cortex._lib import data, exp, models, optimizer
from cortex._lib.data import DatasetPluginBase, DATA_HANDLER
from cortex._lib.models import BuildReference, BuildPluginBase, ModelPluginBase, RoutineReference, RoutinePluginBase
from cortex._lib.handlers import Handler, LossHandler, NetworkHandler, ResultsHandler
from cortex._lib.viz import VizHandler
from cortex._lib.utils import bad_values, update_dict_of_lists


class DatasetPlugin(DatasetPluginBase):
    '''Basic plugin class for datasets into cortex

    '''
    sources = []

    def copy_to_local_path(self, from_path: str) -> str:
        '''Copies data to the local data path.

        Args:
            from_path: Path to data to be copied.

        Returns:
            Path to which data was copied.

        '''
        if from_path.endswith('/'):
            from_path = from_path[:-1]
        basename = path.basename(from_path)
        local_path = self._paths.get('local')

        if local_path is None:
            raise ValueError()
        to_path = path.join(local_path, basename)
        if ((not path.exists(to_path)) and path.exists(from_path)):
            if path.isdir(from_path):
                shutil.copytree(from_path, to_path)
            else:
                shutil.copy(from_path, local_path)

        return to_path

    def add_dataset(self, key, value):
        if key in self._datasets:
            raise KeyError('`{}` already added to datasets in entrypoint'.format(key))
        self._datasets[key] = value

    def get_path(self, source):
        p = self._paths.get(source)
        if p is None:
            raise KeyError('`{}` not found in config.yaml data_paths'.format(source))
        return p

    def set_input_names(self, input_names):
        self._input_names = input_names

    def set_dims(self, **kwargs):
        for k, v in kwargs.items():
            self._dims[k] = v

    def set_scale(self, scale):
        self._scale = scale

    def make_indexing(self, C):
        '''Makes an indexing dataset.

        Index comes in as the last element of the batch.

        Args:
            C: data.Dataset class.

        Returns:
            Wrapped data.Dataset class.

        '''

        class IndexingDataset(C):
            def __getitem__(self, index):
                output = super().__getitem__(index)
                return output + (index,)

        return IndexingDataset


class RoutinePlugin(RoutinePluginBase):
    '''Plugin for custom routines.

    '''
    _protected = ['help', 'kwargs', 'updates']
    _required = ['run']

    plugin_name = None
    plugin_nets = []
    plugin_inputs = []

    def __init__(self, name=None, **kwargs):
        self.updates = 1
        self._names = {}
        self.nets = NetworkHandler()
        self.results = ResultsHandler()
        self.losses = LossHandler(self.nets)
        self.inputs = Handler()
        self.training_nets = []
        self.name = name or self.plugin_name
        self.viz = None

        keys = self.plugin_nets + self.plugin_inputs
        for k, v in kwargs.items():
            if k not in keys:
                raise KeyError('`{}` not supported for this plugin. Available: {}'.format(k, keys))
            if k in self._names:
                raise KeyError('`{}` is already set'.format(k))
            self._names[k] = v

    def __call__(self, **kwargs):
        if not hasattr(self, 'run'):
            raise ValueError('Routine {} does not have `run` method set'.format(self.name))
        self.run(**kwargs)

    def set_viz(self, viz):
        self.viz = viz

    def add_image(self, *args, **kwargs):
        self.viz.add_image(*args, **kwargs)

    def perform_routine(self, **kwargs):
        # Run routine
        if exp.DEVICE == torch.device('cpu'):
            self(**kwargs)
        else:
            with torch.cuda.device(exp.DEVICE.index):
                self(**kwargs)

    def reset(self):
        self.results.clear()
        self.losses.clear()
        self.inputs.clear()


class BuildPlugin(BuildPluginBase):
    '''Plugin for custom build routines.

    '''
    _protected = ['help', 'kwargs']
    _required = ['build']

    plugin_name = None
    plugin_nets = []

    def __init__(self, **kwargs):
        self._data = DATA_HANDLER
        self._nets = models.NETWORK_HANDLER
        self._names = {}

        keys = self.plugin_nets
        for k, v in kwargs.items():
            if k not in keys:
                raise KeyError('`{}` not supported for this plugin. Available: {}'.format(k, keys))
            if k in self._names:
                raise KeyError('`{}` is already set'.format(k))
            self._names[k] = v

    def add_networks(self, **kwargs):
        for k, v in kwargs.items():
            k_ = self._names.get(k, k)
            self._nets[k_] = v

    def get_dims(self, *queries):
        return self._data.get_dims(*queries)

    def add_noise(self, key, dist=None, size=None, **kwargs):
        return self._data.add_noise(key, dist=dist, size=size, **kwargs)

    def __call__(self, **kwargs):
        if not hasattr(self, 'build'):
            raise ValueError('Build {} does not have `build` method set'.format(self.name))
        self.build(**kwargs)


class ModelPlugin(ModelPluginBase):
    _protected = ['help', 'description']
    _required = []
    _optional = ['defaults', 'test_routines', 'finish_train_routines', 'finish_test_routines', 'setup', 'eval_routine']

    plugin_name = None

    def __init__(self):
        self.builds = {}
        self.routines = {}
        self.defaults = {}

        self.train_procedures = []
        self.eval_procedures = []

        self.setup = None

        self.results = ResultsHandler(time=dict(), losses=dict())
        self.nets = models.NETWORK_HANDLER
        self.losses = LossHandler(self.nets)
        self.inputs = Handler()
        self._data = DATA_HANDLER
        self.kwargs = {}

    def add_build(self, build_query, name=None, **kwargs):
        if isinstance(build_query, BuildPlugin):
            build = build_query
        elif inspect.isclass(build_query) and issubclass(build_query, BuildPlugin):
            build = build_query(**kwargs)
        elif isinstance(build_query, str):
            build = BuildReference(build_query, **kwargs)
            name = name or build_query
        else:
            raise TypeError('Unknown build type {}'.format(type(build_query)))
        name = name or build.plugin_name

        if not name in self.builds:
            self.builds[name] = build

    def add_routine(self, routine_query, name=None, **kwargs):
        if isinstance(routine_query, RoutinePlugin):
            routine = routine_query
        elif inspect.isclass(routine_query) and issubclass(routine_query, RoutinePlugin):
            routine = routine_query(**kwargs)
        elif isinstance(routine_query, str):
            routine = RoutineReference(routine_query, **kwargs)
            name = name or routine_query
        else:
            raise TypeError('Unknown routine type {}'.format(type(routine_query)))
        name = name or routine.plugin_name

        if not name in self.routines:
            self.routines[name] = routine

        return name

    def setup_routine_nets(self):
        self.viz = VizHandler()
        for routine in self.routines.values():
            for k in routine.plugin_nets:
                k_ = routine._names.get(k, k)
                routine.nets[k] = self.nets[k_]
            routine.set_viz(self.viz)

    def add_train_procedure(self, *routines, mode='train'):
        routine_names = []
        for routine in routines:
            routine_names.append(self.add_routine(routine))

        self.train_procedures.append((mode, routine_names))

    def add_eval_procedure(self, *routines, mode='test'):
        routine_names = []
        for routine in routines:
            routine_names.append(self.add_routine(routine))

        self.eval_procedures.append((mode, routine_names))

    def run_procedure(self, i, quit_on_bad_values=False):
        self._data.next()
        self.reset_routines()
        mode, procedure = self.train_procedures[i]

        for k, v in self._data.batch.items():
            self.inputs['data.' + k] = v

        for key in procedure:
            routine = self.routines[key]
            kwargs = self.kwargs[key]
            routine.reset()

            receives = routine.plugin_inputs
            sends = [routine._names.get(k, k) for k in receives]
            for send, receive in zip(sends, receives):
                routine.inputs[receive] = self.inputs[send]

            start_time = time.time()
            routine.perform_routine(**kwargs)
            end_time = time.time()

            for loss_key in routine.losses.keys():
                if not loss_key in routine.training_nets:
                    routine.training_nets.append(loss_key)

            routine_losses = dict((k, v.item()) for k, v in routine.losses.items())

            # Check for bad numbers
            bads = bad_values(routine.results)
            if bads and quit_on_bad_values:
                print('Bad values found (quitting): {} \n All:{}'.format(bads, routine.results))
                exit(0)

            # Update results
            update_dict_of_lists(self.results, **routine.results)
            update_dict_of_lists(self.results['losses'], **routine_losses)
            update_dict_of_lists(self.results['time'], **{key: end_time - start_time})

    def train(self, i, quit_on_bad_values=False):
        self._data.next()
        self.reset_routines()
        mode, procedure = self.train_procedures[i]

        for k, v in self._data.batch.items():
            self.inputs['data.' + k] = v

        for key in procedure:
            routine = self.routines[key]
            kwargs = self.kwargs[key]
            routine.reset()

            for k in routine.training_nets:
                k_ = routine._names.get(k, k)
                optimizer.OPTIMIZERS[k_].zero_grad()
                net = routine.nets[k]
                for p in net.parameters():
                    p.requires_grad = True

            receives = routine.plugin_inputs
            sends = [routine._names.get(k, k) for k in receives]
            for send, receive in zip(sends, receives):
                routine.inputs[receive] = self.inputs[send]

            start_time = time.time()
            routine.perform_routine(**kwargs)

            for k, loss in routine.losses.items():
                if loss is not None:
                    loss.backward()
                    k_ = routine._names.get(k, k)
                    optimizer.OPTIMIZERS[k_].step()

            end_time = time.time()

            for loss_key in routine.losses.keys():
                if not loss_key in routine.training_nets:
                    routine.training_nets.append(loss_key)

            routine_losses = dict((k, v.item()) for k, v in routine.losses.items())

            # Check for bad numbers
            bads = bad_values(routine.results)
            if bads and quit_on_bad_values:
                print('Bad values found (quitting): {} \n All:{}'.format(bads, routine.results))
                exit(0)

            # Update results
            update_dict_of_lists(self.results, **routine.results)
            update_dict_of_lists(self.results['losses'], **routine_losses)
            update_dict_of_lists(self.results['time'], **{key: end_time - start_time})


    def reset_routines(self):
        self.losses.clear()
        self.inputs.clear()
        for routine in self.routines.values():
            routine.reset()

    def reset(self):
        self.reset_routines()
        self.results.clear()
        self.results.update(losses=dict(), time=dict())

    def set_train(self):
        for net in self.nets.values():
            net.train()

    def set_eval(self):
        for net in self.nets.values():
            net.eval()


def register_plugin(plugin):
    '''Registers a plugin into c

    Args:
        plugin:

    Returns:

    '''

    if issubclass(plugin, BuildPlugin):
        models.register_build(plugin)
    elif issubclass(plugin, RoutinePlugin):
        models.register_routine(plugin)
    elif issubclass(plugin, ModelPlugin):
        models.register_model(plugin())
    elif issubclass(plugin, DatasetPlugin):
        data.register(plugin)
    else:
        raise ValueError(plugin)