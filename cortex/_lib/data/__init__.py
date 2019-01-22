"""Data module.

This module forms the basic dataset functionality of cortex.

"""

import logging

from .data_handler import DataHandler

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.data')

DATA_HANDLER = DataHandler()
DATASETS = {}
_PLUGINS = {}


def setup(sources: str=None, batch_size=64, n_workers: int=4,
          skip_last_batch: bool=False, inputs=dict(),
          copy_to_local: bool=False, data_args={}, shuffle: bool=True):
    """
    Dataset entrypoint.

    Args:
        sources: Dataset source or dictionary of sources.
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

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)
    DATA_HANDLER.set_input_names(**inputs)

    if sources:
        if not isinstance(sources, dict):
            sources = dict(data=sources)

        for name, source in sources.items():

            plugin = _PLUGINS.get(source, None)
            if plugin is None:
                raise KeyError('Dataset plugin for `{}` not found.'
                               ' Available: {}'
                               .format(source, tuple(_PLUGINS.keys())))

            try:
                plugin.handle(source, copy_to_local=copy_to_local, **data_args)
            except KeyError:
                pass

            DATA_HANDLER.add_dataset(DATASETS, source, name, plugin, n_workers=n_workers,
                                     shuffle=shuffle)

    else:
        raise ValueError('No source provided. Use `--d.source`')


def register(plugin):
    """Registers a dataset plugin into cortex.

    This will make the dataset plugin available to cortex from the command line.

    Args:
        plugin: DatasetPluginBase instance.

    """
    global _PLUGINS
    plugin = plugin()

    for k in plugin.sources:
        if k in _PLUGINS:
            raise KeyError('`{}` already registered in a plugin. '
                           'Try using a different name.'.format(k))
        _PLUGINS[k] = plugin


class DatasetPluginBase:
    """Base class for Dataset Plugin

    This will mostly be unused, see cortex.plugins for the one indended to be used,
    which inherits from this class

    """
    def __init__(self):
        if len(self.sources) == 0:
            raise ValueError('No sources found for dataset plugin.')

        self._dataloader_class = None
