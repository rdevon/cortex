'''Module for plugins

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

__all__ = ['DatasetPlugin', 'ModelPlugin', 'RoutinePlugin', 'BuildPlugin', 'register_plugin']

import inspect
from os import path
import shutil

from torch.utils.data import Dataset

from cortex._lib.data import DatasetPluginBase, register as register_data
from cortex._lib.models import (BuildReference, BuildPluginBase, ModelPluginBase, RoutineReference, RoutinePluginBase,
                                register_build, register_routine, register_model)


class DatasetPlugin(DatasetPluginBase):
    '''Basic plugin class for datasets into cortex

    Attributes:
        sources: list of dataset string names that this plugin will support.

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

    def add_dataset(self, mode: str, dataset: Dataset):
        '''Adds a dataset to the plugin.

        Any dataset added in this way will be used in the training or testing loops, depending on the mode specified.

        Args:
            mode: The data mode that this dataset will be run on.
                `train` and `test` are highly recommended.
            dataset: The dataset object.

        '''
        if mode in self._datasets:
            raise KeyError('`{}` already added to datasets in entrypoint'.format(key))
        self._datasets[mode] = dataset

    def get_path(self, source: str):
        '''Get's the path to a source.

        This is derived from config.yaml file.

        Args:
            source: str for the dataset source.

        Returns:
            The path to the dataset.

        '''
        p = self._paths.get(source)
        if p is None:
            raise KeyError('`{}` not found in config.yaml data_paths'.format(source))
        return p

    def set_input_names(self, input_names):
        '''Sets the names of the elements of the dataset.

        For use downstream in models.

        Args:
            input_names (:obj:`list` of :obj:`str`): The input names.
                Should be the same size as the output of the dataset iterator.

        '''
        self._input_names = input_names

    def set_dims(self, **kwargs):
        ''' Sets the dimenisions of the data

        Args:
            **kwargs: a dictionary of dimension keys and ints.

        '''
        for k, v in kwargs.items():
            self._dims[k] = v

    def set_scale(self, scale):
        '''Sets the min / max values for the data.

        Note:
            This will probably be removed. It doesn't even function right now.

        Args:
            scale (:obj:`tuple` of :obj:`float`): min/max pair.

        '''
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

    Attributes:
        plugin_name (str): Name of the plugin.
        plugin_nets (:obj:`list` of :obj:`str`): Networks that will be used for this routine.
        plugin_inputs (:obj:`list` of :obj:`str`): Inputs that will be used for this routine.
        plugin_outputs (:obj:`list` of :obj:`str`): Outine that for this routine.

    '''
    _protected = ['help', 'kwargs', 'updates']
    _required = ['run']

    plugin_name = None
    plugin_nets = []
    plugin_inputs = []
    plugin_outputs = []
    plugin_optional_inputs = []

    def add_image(self, *args, **kwargs):
        '''Adds image for visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        '''
        self._viz.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        '''Adds histogram for visualizaiton.

        Args:
            *args: TODO
            **kwargs: TODO

        '''
        self._viz.add_histogram(*args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        '''Adds a scatter plot to visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        '''
        self._viz.add_scatter(*args, **kwargs)


class BuildPlugin(BuildPluginBase):
    '''Plugin for custom build routines.

    Attributes:
        plugin_name (str): Name of the plugin.
        plugin_nets (:obj:`list` of :obj:`str`): Networks that will be used for this build.

    '''
    _protected = ['help', 'kwargs']
    _required = ['build']

    plugin_name = None
    plugin_nets = []

    def add_networks(self, **kwargs):
        '''Adds networks to the build.

        Args:
            **kwargs: TODO

        '''
        for k, v in kwargs.items():
            k_ = self._names.get(k, k)
            self._nets[k_] = v

    def get_dims(self, *queries):
        '''Gets dimensions of inputs.

        Args:
            *queries: TODO

        Returns:
            TODO

        '''
        return self._data.get_dims(*queries)

    def add_noise(self, key, dist=None, size=None, **kwargs):
        '''Adds a noise variable to the model.

        Args:
            key (str): Name of the noise variable.
            dist (str): Noise distribution.
            size (int): Size of the noise.
            **kwargs: keyword arguments for noise distribution.
        '''
        self._data.add_noise(key, dist=dist, size=size, **kwargs)


class ModelPlugin(ModelPluginBase):
    '''Module plugin.

    Attributes:
        plugin_name (str): Name of the plugin.
        data_defaults (:obj:`dict`): Data defaults.
        train_defaults (:obj:`dict`): Train defaults.
        optimizer_defaults (:obj:`dict`): Optimizer defaults.

    '''
    _protected = ['help', 'description']
    _required = []
    _optional = ['setup']

    plugin_name = None
    data_defaults = {}
    train_defaults = {}
    optimizer_defaults = {}

    def add_build(self, build_query, name=None, **kwargs):
        '''Adds a build plugin.

        Args:
            build_query: Build plugin.
                Can be a string, a BuildPlugin instance, or BuildPlugin class
            name (str, optional): Name for the build.
                If not set, `BuildPlugin.plugin_name` will be used.
            **kwargs: TODO

        '''
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
        '''Adds a routine

        Args:
            routine_query: Routine plugin.
                Can be a string, a RoutinePlugin instance, or RoutinePlugin class
            name: (str, optional): Name for the routine.
                If not set, `RoutinePlugin.plugin_name` will be used.
            **kwargs: TODO

        '''
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

    def add_train_procedure(self, *routines, mode: str='train', updates_per_routine=None):
        '''Adds a training procedure.

        Args:
            *routines: TODO
            mode (str): Data mode on which the procedure will be run.
            updates_per_routine (:obj:`list` of :obj:`int`) Dictionary of updates for each routine.

        '''
        updates_per_routine = updates_per_routine or [1 for _ in routines]
        if len(routines) != len(updates_per_routine):
            raise ValueError('Number of routines must match number of updates.')
        routine_names = []
        for routine in routines:
            routine_names.append(self.add_routine(routine))

        self.train_procedures.append((mode, routine_names, updates_per_routine))

    def add_eval_procedure(self, *routines, mode='test'):
        '''Adds a evaluation procedure.

                Args:
                    *routines: TODO
                    mode (str): Data mode on which the procedure will be run.

                '''
        routine_names = []
        for routine in routines:
            routine_names.append(self.add_routine(routine))

        self.eval_procedures.append((mode, routine_names))


def register_plugin(plugin):
    '''Registers a plugin into cortex

    Args:
        plugin: TODO

    Returns:

    '''

    if issubclass(plugin, BuildPlugin):
        register_build(plugin)
    elif issubclass(plugin, RoutinePlugin):
        register_routine(plugin)
    elif issubclass(plugin, ModelPlugin):
        register_model(plugin())
    elif issubclass(plugin, DatasetPlugin):
        register_data(plugin)
    else:
        raise ValueError(plugin)