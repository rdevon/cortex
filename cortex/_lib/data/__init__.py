'''Data module

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
from os import path
import shutil

from .data_handler import DataHandler
from ..config import CONFIG


logger = logging.getLogger('cortex.data')

DATA_HANDLER = DataHandler()

_args = dict(
    source=None,
    batch_size=64,
    n_workers=4,
    skip_last_batch=False,
    copy_to_local=False,
    data_args={},
)

_args_help = dict(
    source='Dataset (location (full path) or name).',
    batch_size='Batch size',
    n_workers='Number of workers',
    skip_last_batch='Skip the last batch of the epoch',
    copy_to_local='Copy data to local path',
    data_args='Args for the data. Set by the user.'
)


def setup(source, batch_size=64, n_workers: int=4, skip_last_batch: bool=False,
          DataLoader=None, copy_to_local: bool=False, data_args={}, shuffle: bool=True):
    '''Sets up the datasets.

    Args:
        source: Dataset source or list of sources.
        batch_size: Batch size or dict of batch sizes.
        noise_variables: Dict of noise variables.
        n_workers: Number of workers for DataLoader class.
        skip_last_batch: Whether to skip the last batch if the size is smaller than batch_size.
        DataLoader: Optional user-defined DataLoader.
        copy_to_local: Copy the data to a local path.
        data_args: Arguments for dataset plugin
        shuffle:

    '''
    global DATA_HANDLER

    if source and not isinstance(source, (list, tuple)):
        sources = [source]
    else:
        sources = source

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)

    if sources:
        for source in sources:
            plugin = _PLUGINS.get(source, None)
            if plugin is None:
                raise KeyError('Dataset plugin for `{}` not found'.format(source))

            plugin.handle(source, copy_to_local=copy_to_local, **data_args)
            DATA_HANDLER.add_dataset(source, plugin, n_workers=n_workers, shuffle=shuffle, DataLoader=DataLoader)
    else:
        raise ValueError('No source provided. Use `--d.source`')


_PLUGINS = dict()
class DatasetPlugin():
    '''Basic plugin class for datasets into cortex

    '''
    _sources = []

    def __init__(self):

        if len(self._sources) == 0:
            raise ValueError('No sources found for dataset entry point.')

        for k in self._sources:
            self._register(k)

        self.datasets = {}
        self.dims = {}
        self.input_names = None
        self._scale = None
        if CONFIG.data_paths is None:
            raise ValueError('`data_paths` not set in config.')
        self._paths = CONFIG.data_paths

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

        local_path = CONFIG.data_paths.get('local')

        if local_path is None:
            raise ValueError()
        to_path = path.join(local_path, basename)
        if ((not path.exists(to_path)) and path.exists(from_path)):
            logger.info('Copying {} to {}'.format(from_path, to_path))
            if path.isdir(from_path):
                shutil.copytree(from_path, to_path)
            else:
                shutil.copy(from_path, local_path)

        return to_path

    def add_dataset(self, key, value):
        if key in self.datasets:
            raise KeyError('`{}` already added to datasets in entrypoint'.format(key))
        self.datasets[key] = value

    def get_path(self, source):
        p = self._paths.get(source)
        if p is None:
            raise KeyError('`{}` not found in config.yaml data_paths'.format(source))
        return p

    def set_input_names(self, input_names):
        self.input_names = input_names

    def set_dims(self, **kwargs):
        for k, v in kwargs.items():
            self.dims[k] = v

    def set_scale(self, scale):
        self.scale = scale

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

    def _register(self, dataset_key):
        global _PLUGINS
        if dataset_key in _PLUGINS:
            raise KeyError('`{}` already registered in a plugin. '
                           'Try using a different name.'.format(dataset_key))
        _PLUGINS[dataset_key] = self
