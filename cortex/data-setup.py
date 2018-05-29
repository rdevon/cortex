"""
TODO
"""
from src.cortex.core.DataHandler import DataHandler
import logging
from . import experiment
DATA_HANDLER = DataHandler()
LOGGER = logging.getLogger('cortex.data_setup')

def setup(
        source=None,
        batch_size=None,
        noise_variables=None,
        n_workers=None,
        skip_last_batch=None,
        test_on_train=None,
        Dataset=None,
        DataLoader=None,
        transform=None,
        copy_to_local=False,
        duplicate=None,
        transform_args={}):

    global DATA_HANDLER, NOISE

    if source and not isinstance(source, (list, tuple)):
        source = [source]

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)

    if DataLoader is not None:
        LOGGER.info('Loading custom DataLoader class, {}'.format(DataLoader))
        experiment.ARGS['data']['DataLoader'] = DataLoader

    if Dataset is not None:
        LOGGER.info('Loading custom Dataset class, {}'.format(Dataset))
        experiment.ARGS['data']['Dataset'] = Dataset

    if transform is not None:
        LOGGER.info('Loading custom transform function, {}'.format(transform))
        experiment.ARGS['data']['transform'] = transform

    if source:
        for source_ in source:
            DATA_HANDLER.add_dataset(
                source_,
                test_on_train,
                n_workers=n_workers,
                DataLoader=DataLoader,
                Dataset=Dataset,
                transform=transform,
                transform_args=transform_args,
                duplicate=duplicate,
                copy_to_local=copy_to_local)
    else:
        raise ValueError('No source provided. Use `--d.source`')

    if noise_variables:
        for k, v in noise_variables.items():
            DATA_HANDLER.add_noise(k, **v)
