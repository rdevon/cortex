'''Utility methods

'''

import argparse
import logging
import os

import numpy as np


logger = logging.getLogger('cortex.util')

try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1


def print_section(s):
    '''For printing sections to scripts nicely.
    Args:
        s (str): string of section
    '''
    h = s + ('-' * (_columns - len(s)))
    print h


def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str, help='Architecture name')
    parser.add_argument('-o', '--out_path', default=None,
                        help=('Output path directory. All model results will go'
                              ' here. If a new directory, a new one will be '
                              'created, as long as parent exists.'))
    parser.add_argument('-n', '--name', default=None,
                        help=('Name of the experiment. If given, base name of '
                              'output directory will be `--name`. If not given,'
                              ' name will be the base name of the `--out_path`')
                        )
    parser.add_argument('-r', '--reload', type=str, default=None,
                        help=('Path to model to reload.'))
    parser.add_argument('-a', '--args', default=None, type=str,
                        help=('Arguments for the main file'))
    parser.add_argument('-S', '--source', type=str, default=None,
                        help='Dataset (location (full path) or name).')
    parser.add_argument('-m', '--meta', type=str, default=None)
    parser.add_argument('-c', '--config_file', default=None,
                        help=('Configuration yaml file. '
                              'See `exps/` for examples'))
    parser.add_argument('-k', '--clean', action='store_true', default=False,
                        help=('Cleans the output directory. '
                              'This cannot be undone!'))
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser


def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]


def bad_values(d):
    failed = {}
    for k, v in d.items():
        if np.isnan(v) or np.isinf(v):
            failed[k] = v

    if len(failed) == 0:
        return False
    return failed