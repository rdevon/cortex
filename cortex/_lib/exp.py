'''Experiment module.

Used for saving, loading, summarizing, etc

'''

import copy
import logging
import os
from os import path
from shutil import copyfile, rmtree
import yaml

import torch

from cortex._lib.results import Results
from cortex._lib.log_utils import set_file_logger

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.exp')

# Experiment info
NAME = 'X'
OUT_DIRS = dict()
ARGS = dict(data=dict(), model=dict(), optimizer=dict(), train=dict(), viz=dict())
RESULTS = Results()
INFO = dict(name=NAME, epoch=0, data_steps=-1)
DEVICE = torch.device('cpu')
DEVICE_IDS = None


def _file_string(prefix: str = '') -> str:
    if prefix == '':
        return NAME
    return '{}_{}'.format(NAME, prefix)


def reset():
    global INFO, ARGS
    INFO.update(name=NAME, epoch=0, data_steps=-1)
    RESULTS.clear()
    ARGS.update(data=dict(), model=dict(), optimizer=dict(), train=dict(), viz=dict())


def from_yaml(config_file: str = None) -> dict:
    '''Loads hyperparameters from a yaml file.

    Args:
        config_file: Config file path.

    Returns:
        dict: Dictionary of hyperparameters

    '''

    if config_file is not None:
        with open(config_file, 'r') as f:
            d = yaml.load(f)
        logger.info('Loading config {}'.format(d))
        return d

    else:
        return {}


def reload_model(model_to_reload: str):
    '''

    Args:
        model_to_reload: Path to model to reload.

    '''
    if not path.isfile(model_to_reload):
        raise ValueError('Cannot find {}'.format(model_to_reload))

    logger.info('Reloading from {} and creating backup'.format(model_to_reload))
    copyfile(model_to_reload, model_to_reload + '.bak')

    return torch.load(model_to_reload, map_location='cpu')


def save(model, prefix: str = ''):
    '''Saves a model.

    Args:
        model: Model to save.
        prefix: Prefix for the save file.

    '''
    try:
        filename = _file_string(prefix)
        binary_dir = OUT_DIRS.get('binary_dir', None)
        if binary_dir is None:
            return

        def strip_Nones(d):
            d_ = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    d_[k] = strip_Nones(v)
                elif v is not None:
                    d_[k] = v
            return d_

        nets = {}
        for k, net in model._all_nets.items():
            nets[k] = copy.deepcopy(net.module).to('cpu')

        state = dict(
            nets=nets,
            info=INFO,
            args=ARGS,
            out_dirs=OUT_DIRS,
            results=RESULTS.todict()
        )

        file_path = path.join(binary_dir, '{}.t7'.format(filename))
        if prefix == 'last':
            try:
                copyfile(file_path, file_path + '.bak')
            except FileNotFoundError:
                pass

        logger.info('Saving checkpoint {}'.format(file_path))
        torch.save(state, file_path)
    except OSError as e:
        logger.error('Save failed, skipping: {}'.format(e))


def save_best(model, train_results, best, save_on_best, save_on_lowest):
    '''Saves the best model according to some metric.

    Args:
        model (ModelPlugin): Model.
        train_results (dict): Dictionary of results from training data.
        best (float): Last best value.
        save_on_best (bool): If true, save when best is found.
        save_on_lowest (bool): If true, lower is better.

    Returns:
        float: the best value.

    '''
    flattened_results = {}
    for k, v in train_results.items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                flattened_results[k + '.' + k_] = v_
        else:
            flattened_results[k] = v
    if save_on_best in flattened_results:
        # TODO(Devon) This needs to be fixed.
        # when train_for is set, result keys vary per epoch
        # if save_on_best not in flattened_results:
        #    raise ValueError('`save_on_best` key `{}` not found.
        #  Available: {}'.format(
        #        save_on_best, tuple(flattened_results.keys())))
        current = flattened_results[save_on_best]
        if not best:
            found_best = True
        elif save_on_lowest:
            found_best = current < best
        else:
            found_best = current > best
        if found_best:
            best = current
            print(
                '\nFound best {} (train): {}'.format(
                    save_on_best, best))
            save(model, prefix='best_' + save_on_best)

        return best


def setup_out_dir(out_path: str, global_out_path: str, name: str = None, clean: bool = False):
    '''Sets up the output directory of an experiment.

    Args:
        out_path: Output directory.
        global_out_path: Global path for cortex.
        name: Name of experiment.
        clean: Clean the experiment directory, if exists.

    '''
    global OUT_DIRS

    if out_path is None:
        if name is None:
            raise ValueError('If `out_path` (-o) argument is not set, you '
                             'must set the `name` (-n)')
        out_path = global_out_path
        if out_path is None:
            raise ValueError('If `--out_path` (`-o`) argument is not set, you '
                             'must set both the name argument and configure '
                             'the out_path entry in `config.yaml`')

    if name is not None:
        out_path = path.join(out_path, name)

    if not path.isdir(out_path):
        logger.info('Creating out path `{}`'.format(out_path))
        os.mkdir(out_path)

    binary_dir = path.join(out_path, 'binaries')
    image_dir = path.join(out_path, 'images')

    if clean:
        logger.warning('Cleaning directory (cannot be undone)')
        if path.isdir(binary_dir):
            rmtree(binary_dir)
        if path.isdir(image_dir):
            rmtree(image_dir)

    if not path.isdir(binary_dir):
        os.mkdir(binary_dir)
    if not path.isdir(image_dir):
        os.mkdir(image_dir)

    logger.info('Setting out path to `{}`'.format(out_path))
    logger.info('Logging to `{}`'.format(path.join(out_path, 'out.log')))
    set_file_logger(path.join(out_path, 'out.log'))

    OUT_DIRS.update(binary_dir=binary_dir, image_dir=image_dir)


def setup_device(device: [int] or int):
    global DEVICE, DEVICE_IDS

    if isinstance(device, int):
        device = [device]
    DEVICE_IDS = device
    device = 'cuda:{}'.format(device[0])
    if torch.cuda.is_available() and device != 'cpu':
        DEVICE = torch.device(device)
    else:
        logger.info('Using CPU')
