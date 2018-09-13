"""Module for plugins"""

import logging
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

logger = logging.getLogger('cortex.plugins')


class DatasetPlugin(DatasetPluginBase):
    """Basic plugin class for datasets into cortex.

    Attributes:
        sources: list of dataset string names that this plugin will support.

    """
    sources = []

    def copy_to_local_path(self, from_path: str) -> str:
        """ Copies data to a local path.

        Path is set in the .cortex.yml file. This can be set up through
        `cortex setup`.

        Args:
            from_path: The path to the data to be copied.

        """
        if from_path.endswith('/'):
            from_path = from_path[:-1]
        basename = path.basename(from_path)
        local_path = CONFIG.data_paths.get('local')

        if local_path is None:
            raise KeyError(
                '`{}` not found in {} data_paths'
                .format(local_path, _config_name))
        to_path = path.join(local_path, basename)
        if (not path.exists(to_path)) and path.exists(from_path):

            logger.info('Copying dataset {} from {} to {} directory.... '
                        '(This may take time)'
                        .format(self.__class__.__name__, from_path, to_path))

            if path.isdir(from_path):
                shutil.copytree(from_path, to_path)
            else:
                shutil.copy(from_path, local_path)

            logger.info('Finished copying.')

        return to_path

    def add_dataset(self, mode: str, dataset: Dataset):
        """Adds a dataset to the plugin.

        Any dataset added in this way will be used in the training or testing
        loops, depending on the mode specified.

        Args:
            mode: The data mode that this dataset will be run on.
                `train` and `test` are highly recommended.
            dataset: The dataset object.

        """
        if mode in self._datasets:
            raise KeyError(
                '`{}` already added to datasets in entrypoint'.format(mode))
        self._datasets[mode] = dataset

    def set_dataloader_class(self, dataloader_class: type):
        """Set dataloader class.

         It will be used instead of :class:`torch.utils.data.DataLoader`.

        Args:
            dataloader_class: custom data loader class

        Notes:
            This method can be used to pass custom collate function, pass
            arguments to the dataloader or use a completely custom loader.

        Example:
            ```
            class MyData(DatasetPlugin):
                sources = ['MyData']

                def handle(self, source, copy_to_local=False, normalize=True,
                           tanh_normalization=False, **transform_args):
                    train_set = ...

                    def collate(batch):
                        collated_batch = ...
                        return collated_batch

                    NewDataLoader = partial(DataLoader, collate_fn=collate)
                    self.add_dataset('train', train_set)
                    self.set_dataloader_class(dataloader_class=NewDataLoader)
            ```

        """
        self._dataloader_class = dataloader_class

    def get_path(self, source: str):
        """Get's the path to a source.

        This is derived from config.yaml file.

        Args:
            source: str for the dataset source.

        Returns:
            The path to the dataset.

        """
        p = CONFIG.data_paths.get(source)
        if p is None:
            raise KeyError(
                '`{}` not found in {} data_paths'.format(source, _config_name))
        return p

    def set_input_names(self, input_names):
        """Sets the names of the elements of the dataset.

        For use downstream in models.

        Args:
            input_names (:obj:`list` of :obj:`str`): The input names.
                Should be the same size as the output of the dataset iterator.

        """
        self._input_names = input_names

    def set_dims(self, **kwargs):
        """Sets the dimensions of the data.

        Args:
            **kwargs: a dictionary of dimension keys and ints.

        """
        for k, v in kwargs.items():
            self._dims[k] = v

    def set_scale(self, scale):
        """Sets the min / max values for the data.

        Note:
            This will probably be removed. It doesn't even function right now.

        Args:
            scale (:obj:`tuple` of :obj:`float`): min/max pair.

        """
        self._scale = scale

    def make_indexing(self, C):
        """Makes an indexing dataset.

        Index comes in as the last element of the batch.

        Args:
            C: data.Dataset class.

        Returns:
            Wrapped data.Dataset class.

        """

        class IndexingDataset(C):
            def __getitem__(self, index):
                output = super().__getitem__(index)
                return output + (index,)

        return IndexingDataset


class ModelPlugin(ModelPluginBase):
    """Module plugin.

    Attributes:
        plugin_name (str): Name of the plugin.
        data_defaults (:obj:`dict`): Data defaults.
        train_defaults (:obj:`dict`): Train defaults.
        optimizer_defaults (:obj:`dict`): Optimizer defaults.

    """
    _protected = ['description']
    _required = []
    _optional = ['setup']

    defaults = {}

    def build(self, *args, **kwargs):
        """Builds the neural networks.

        The the model is to build something, this needs to be overridden.

        Args:
            *args: Inputs to be passed to the function.
            **kwargs: Hyperparameters to be passed to the function

        """
        raise NotImplementedError(
            '`build` is not implemented for model class {}'
            .format(self.__class__.__name__))

    def routine(self, *args, **kwargs):
        """Derives losses and results.

        The the model is to train something, this needs to be
        overridden.

        Args:
            *args: Inputs to be passed to the function.
            **kwargs: Hyperparameters to be passed to the function

        """
        raise NotImplementedError(
            '`routine` is not implemented for model class {}'
            .format(self.__class__.__name__))

    def visualize(self, *args, **kwargs):
        """Visualizes.

        The the model is to visualize something, this needs to be
        overridden.

        Args:
            *args: Inputs to be passed to the function.
            **kwargs: Hyperparameters to be passed to the function

        """
        pass

    def train_step(self):
        """Makes a training step.

        This can be overridden to change the behavior at each training step.

        """
        self.data.next()
        self.routine(auto_input=True)
        self.optimizer_step()

    def eval_step(self):
        """Makes an evaluation step.

        This can be overridden to change the behavior of each evaluation step.

        """
        self.data.next()
        self.routine(auto_input=True)

    def optimizer_step(self):
        """Makes a step of the optimizers for which losses are defined.

        This can be overridden to change the behavior of the optimizer.

        """
        keys = self.losses.keys()

        for i, k in enumerate(keys):
            loss = self.losses.pop(k)
            loss.backward(retain_graph=(i < len(keys)))
            #  TODO(Devon): Is this a good idea?
            key = self.nets._aliases.get(k, k)

            optimizer = self._optimizers.get(key)
            if optimizer is not None:
                optimizer.step()

    def train_loop(self):
        """The training loop.

        This can be overridden to change the behavior of the training loop.

        """

        try:
            while True:
                self.train_step()

        except StopIteration:
            pass

    def eval_loop(self):
        """The evaluation loop.

        This can be overridden to change the behavior of the evaluation loop.

        """

        try:
            while True:
                self.eval_step()

        except StopIteration:
            pass

    def get_dims(self, *queries):
        """Gets dimensions of inputs.

        Args:
            *queries: Variables to get dimensions of .

        Returns:
            Dimensions of the variables.

        """
        return self._data.get_dims(*queries)

    def add_noise(self, key, dist=None, size=None, **kwargs):
        """Adds a noise variable to the model.

        Args:
            key (str): Name of the noise variable.
            dist (str): Noise distribution.
            size (int): Size of the noise.
            **kwargs: keyword arguments for noise distribution.
        """
        self._data.add_noise(key, dist=dist, size=size, **kwargs)

    def add_image(self, *args, **kwargs):
        """Adds image for visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self._viz.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """Adds histogram for visualizaiton.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self._viz.add_histogram(*args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        """Adds a scatter plot to visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self._viz.add_scatter(*args, **kwargs)


def register_plugin(plugin):
    """Registers a plugin into cortex

    Args:
        plugin: TODO

    Returns:

    """

    if issubclass(plugin, ModelPlugin):
        register_model(plugin)
    elif issubclass(plugin, DatasetPlugin):
        register_data(plugin)
    else:
        raise ValueError(plugin)
