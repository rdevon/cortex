'''Module for plugins

'''

from os import path
import shutil

from torch.utils.data import Dataset

from cortex._lib.config import CONFIG, _config_name
from cortex._lib.data import DatasetPluginBase, register as register_data
from cortex._lib.models import ModelPluginBase, register_model

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

__all__ = [
    'DatasetPlugin',
    'ModelPlugin',
    'register_plugin']


class DatasetPlugin(DatasetPluginBase):
    '''Basic plugin class for datasets into cortex

    Attributes:
        sources: list of dataset string names that this plugin will support.

    '''

    sources = []

    def copy_to_local_path(self, from_path: str) -> str:
        if from_path.endswith('/'):
            from_path = from_path[:-1]
        basename = path.basename(from_path)
        local_path = CONFIG.data_paths.get('local')

        if local_path is None:
            raise KeyError(
                '`{}` not found in {} data_paths'
                .format(local_path, _config_name))
        to_path = path.join(local_path, basename)
        if ((not path.exists(to_path)) and path.exists(from_path)):
            if path.isdir(from_path):
                shutil.copytree(from_path, to_path)
            else:
                shutil.copy(from_path, local_path)

        return to_path

    def add_dataset(self, mode: str, dataset: Dataset):
        '''Adds a dataset to the plugin.

        Any dataset added in this way will be used in the training or testing
        loops, depending on the mode specified.

        Args:
            mode: The data mode that this dataset will be run on.
                `train` and `test` are highly recommended.
            dataset: The dataset object.

        '''
        if mode in self._datasets:
            raise KeyError(
                '`{}` already added to datasets in entrypoint'.format(mode))
        self._datasets[mode] = dataset

    def get_path(self, source: str):
        '''Get's the path to a source.

        This is derived from config.yaml file.

        Args:
            source: str for the dataset source.

        Returns:
            The path to the dataset.

        '''
        p = CONFIG.data_paths.get(source)
        if p is None:
            raise KeyError(
                '`{}` not found in {} data_paths'.format(source, _config_name))
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


class ModelPlugin(ModelPluginBase):
    '''Module plugin.

    Attributes:
        plugin_name (str): Name of the plugin.
        data_defaults (:obj:`dict`): Data defaults.
        train_defaults (:obj:`dict`): Train defaults.
        optimizer_defaults (:obj:`dict`): Optimizer defaults.

    '''
    _protected = ['description']
    _required = []
    _optional = ['setup']

    plugin_name = None
    data_defaults = {}
    train_defaults = {}
    optimizer_defaults = {}

    def get_dims(self, *queries):
        '''Gets dimensions of inputs.

        Args:
            *queries: TODO

        Returns:
            TODO

        '''
        return self._data.get_dims(*queries)

    def build_networks(self):
        for k, build in self.builds.items():
            kwargs = self.get_kwargs(build)
            build(**kwargs)

    def add_noise(self, key, dist=None, size=None, **kwargs):
        '''Adds a noise variable to the model.

        Args:
            key (str): Name of the noise variable.
            dist (str): Noise distribution.
            size (int): Size of the noise.
            **kwargs: keyword arguments for noise distribution.
        '''
        self._data.add_noise(key, dist=dist, size=size, **kwargs)

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

    def _add_routine_name(self, routine):
        '''Adds a routine

        Args:
            routine: Routine plugin instance.

        '''

        if routine in self.routines.values():
            return routine.name

        name = routine.plugin_name
        self.routines[name] = routine

        return name

    def add_train_procedure(self, *routines, mode: str='train',
                            updates_per_routine=None):
        '''Adds a training procedure.

        Args:
            *routines: TODO
            mode (str): Data mode on which the procedure will be run.
            updates_per_routine (:obj:`list` of :obj:`int`) Dictionary
            of updates for each routine.

        '''
        updates_per_routine = updates_per_routine or [1 for _ in routines]
        if len(routines) != len(updates_per_routine):
            raise ValueError(
                'Number of routines must match number of updates.')
        routine_names = []
        for routine in routines:
            routine_names.append(self._add_routine_name(routine))

        self._train_procedures.append(
            (mode, routine_names, updates_per_routine))

    def add_eval_procedure(self, *routines, mode='test'):
        '''Adds a evaluation procedure.

        Args:
            *routines: TODO
            mode (str): Data mode on which the procedure will be run.

        '''
        routine_names = []
        for routine in routines:
            routine_names.append(self._add_routine_name(routine))

        self._eval_procedures.append((mode, routine_names))


def register_plugin(plugin):
    '''Registers a plugin into cortex

    Args:
        plugin: TODO

    Returns:

    '''

    if issubclass(plugin, ModelPlugin):
        register_model(plugin())
    elif issubclass(plugin, DatasetPlugin):
        register_data(plugin)
    else:
        raise ValueError(plugin)
