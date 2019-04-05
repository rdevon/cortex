"""Module for plugins

"""

import logging
from os import path
import shutil
import time

from cortex._lib.config import CONFIG, _config_name
from cortex._lib.data import DatasetPluginBase, register as register_data, DATASETS
from cortex._lib.models import ModelPluginBase, register_model

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

__all__ = [
    'DatasetPlugin',
    'ModelPlugin',
    'register_plugin']

logger = logging.getLogger('cortex.plugins')


class DatasetPlugin(DatasetPluginBase):
    """Basic plugin class for datasets into cortex

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
        if ((not path.exists(to_path)) and path.exists(from_path)):

            logger.info('Copying dataset {} from {} to {} directory.... '
                        '(This may take time)'
                        .format(self.__class__.__name__, from_path, to_path))

            if path.isdir(from_path):
                shutil.copytree(from_path, to_path)
            else:
                shutil.copy(from_path, local_path)

            logger.info('Finished copying.')

        return to_path

    def add_dataset(self, source: str, data=None, input_names=None, scale=None, dims=None):
        """Adds a dataset to the plugin.

        Any dataset added in this way will be used in the training or testing
        loops, depending on the mode specified.

        Args:
            source: The name of the dataset.
            data: dictionary of Dataset objects.
            input_names: list of input name strings.
            scale: scaling parameter for visualization.
            dims: dictionary of dimensions of data.

        """
        if source in DATASETS:
            raise KeyError(
                '`{}` already added to datasets'.format(source))
        DATASETS[source] = dict(data=data, input_names=input_names, scale=scale, dims=dims)

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

    def optimizer_step(self, retain_graph=False):
        """Makes a step of the optimizers for which losses are defined.

        This can be overridden to change the behavior of the optimizer.

        """
        keys = self.losses.keys()

        def op_step(key, loss, retain_graph=False):
            optimizer = self._optimizers.get(key)

            if optimizer is not None:
                optimizer.zero_grad()

                start = time.time()
                loss.backward(retain_graph=retain_graph)

                self.add_grads(**{key: optimizer.grad_stats()})
                optimizer.step()
                end = time.time()
                self.times['Optimizer {}'.format(key)] = end - start

        if 'ALL' in self._optimizers.keys():  # One optimizer
            loss = sum(self.losses.values())
            op_step('ALL', loss)
        else:
            for i, k in enumerate(keys):
                key = self.nets._aliases.get(k, k)
                loss = self.losses.get(k)
                op_step(key, loss, retain_graph=(i + 1 < len(keys)))

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
        queries = self._map_data_queries(*queries)
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
        self.viz.add_image(*args, **kwargs)

    def add_histogram(self, *args, **kwargs):
        """Adds histogram for visualizaiton.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self.viz.add_histogram(*args, **kwargs)

    def add_scatter(self, *args, **kwargs):
        """Adds a scatter plot to visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self.viz.add_scatter(*args, **kwargs)

    def add_heatmap(self, *args, **kwargs):
        """Adds a heatmap to visualization.

        Args:
            *args: TODO
            **kwargs: TODO

        """
        self.viz.add_heatmap(*args, **kwargs)


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
