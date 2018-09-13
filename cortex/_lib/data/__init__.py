"""Data module"""

import logging

from .data_handler import DataHandler

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.data')

DATA_HANDLER = DataHandler()
_PLUGINS = {}


def setup(source: str=None, batch_size=64, n_workers: int=4,
          skip_last_batch: bool=False, inputs=dict(),
          copy_to_local: bool=False, data_args={}, shuffle: bool=True):
    """
    Dataset entrypoint.

    Args:
        source: Dataset source or list of sources.
        batch_size: Batch size or dict of batch sizes.
        noise_variables: Dict of noise variables.
        n_workers: Number of workers for DataLoader class.
        skip_last_batch: Whether to skip the last batch if the size
            is smaller than batch_size.
        inputs: Dictionary of input mappings.
        copy_to_local: Copy the data to a local path.
        data_args: Arguments for dataset plugin.
        shuffle: Shuffle the dataset.

    """
    global DATA_HANDLER

    if source and not isinstance(source, (list, tuple)):
        sources = [source]
    else:
        sources = source

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)
    DATA_HANDLER.set_inputs(**inputs)

    if sources:
        for source in sources:
            # TODO: Hardcoded for testing purpose.
            if not isinstance(source, str):
                source = 'CIFAR10'
            plugin = _PLUGINS.get(source, None)
            if plugin is None:
                raise KeyError('Dataset plugin for `{}` not found.'
                               ' Available: {}'
                               .format(source, tuple(_PLUGINS.keys())))

            plugin.handle(source, copy_to_local=copy_to_local, **data_args)
            DATA_HANDLER.add_dataset(source, plugin, n_workers=n_workers,
                                     shuffle=shuffle)
    else:
        raise ValueError('No source provided. Use `--d.source`')


def register(plugin):
    global _PLUGINS
    plugin = plugin()

    for k in plugin.sources:
        if k in _PLUGINS:
            raise KeyError('`{}` already registered in a plugin. '
                           'Try using a different name.'.format(k))
        _PLUGINS[k] = plugin


class DatasetPluginBase:
    def __init__(self):
        if len(self.sources) == 0:
            raise ValueError('No sources found for dataset entry point.')

        self._datasets = {}
        self._dims = {}
        self._input_names = None
        self._scale = None
        self._dataloader_class = None
