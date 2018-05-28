'''Data module

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
from os import path
import shutil

from . import torchvision_datasets
from .data_handler import DataHandler
from .. import CONFIG


logger = logging.getLogger('cortex.data')

DATA_HANDLER = DataHandler()

_args = dict(
    source=None,
    batch_size=64,
    n_workers=4,
    skip_last_batch=False,
    copy_to_local=False,
    transform_args={},
)

_args_help = dict(
    source='Dataset (location (full path) or name).',
    batch_size='Batch size',
    n_workers='Number of workers',
    skip_last_batch='Skip the last batch of the epoch',
    copy_to_local='Copy data to local path',
    transform_args='Transformation args for the data. Keywords: normalize (bool), center_crop (int), '
                   'image_size (int or tuple), random_crop (int), use_sobel (bool), random_resize_crop (int), or flip (bool)',
)


def setup(source, batch_size=64, n_workers: int=4, skip_last_batch: bool=False,
          DataLoader=None, transform=None, copy_to_local: bool=False,
          transform_args={}, shuffle: bool=True):
    '''Sets up the datasets.

    Args:
        source: Dataset source or list of sources.
        batch_size: Batch size or dict of batch sizes.
        noise_variables: Dict of noise variables.
        n_workers: Number of workers for DataLoader class.
        skip_last_batch: Whether to skip the last batch if the size is smaller than batch_size.
        Dataset: Optional user-defined Dataset class.
        DataLoader: Optional user-defined DataLoader.
        transform: Optional user-defined transform function.
        copy_to_local: Copy the data to a local path.
        transform_args: Arguments for transform function
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
            entrypoint = _PLUGINS.get(source, None)
            if entrypoint is None:
                raise KeyError('Entrypoint for `{}` not found'.format(source))

            entrypoint.handle(source, copy_to_local=copy_to_local, transform=transform, **transform_args)
            DATA_HANDLER.add_dataset(source, entrypoint, n_workers=n_workers, shuffle=shuffle, DataLoader=DataLoader)
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
        if CONFIG.data_paths is None:
            raise ValueError('`data_paths` not set in config.')
        self.paths = CONFIG.data_paths

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

        local_path = CONFIG.data_path.get('local')

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

    def set_input_names(self, input_names):
        self.input_names = input_names

    def set_dims(self, **kwargs):
        for k, v in kwargs.items():
            self.dims[k] = v

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
