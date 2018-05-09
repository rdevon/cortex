'''Utility methods

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import argparse
import logging
import os

import numpy as np
import torch

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
    print(h)


_protected_args = ['arch', 'out_path', 'name', 'reload', 'args', 'source', 'copy_to_local', 'meta', 'config_file',
                  'clean', 'verbosity', 'test']

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
    parser.add_argument('-R', '--reloads', type=str, nargs='+', default=None)
    parser.add_argument('-a', '--args', default=None, type=str,
                        help=('Arguments for the main file'))
    parser.add_argument('-S', '--source', type=str, default=None,
                        help='Dataset (location (full path) or name).')
    parser.add_argument('-C', '--copy_to_local', action='store_true', default=False)
    parser.add_argument('-m', '--meta', type=str, default=None)
    parser.add_argument('-c', '--config_file', default=None,
                        help=('Configuration yaml file. '
                              'See `exps/` for examples'))
    parser.add_argument('-k', '--clean', action='store_true', default=False,
                        help=('Cleans the output directory. '
                              'This cannot be undone!'))
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    parser.add_argument('-t', '--test', action='store_true', default=False)
    return parser


class Handler(dict):
    '''
    Simple dict-like container with support for `.` access
    Note: some of the functionalty might not work correctly as a dict, but so far simple tests pass.
    '''

    __delattr__ = dict.__delitem__
    _protected = dir({})
    _type = None
    _get_error_string = 'Keyword `{}` not found (add as a dict entry). Found: {}'


    def check_key_value(self, k, v):
        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. Only `{}` and subclasses are supported'.format(
                type(v), k, self._type))

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().__init__(**kwargs)

    def __setitem__(self, k, v):
        self.check_key_value(k, v)
        super().__setitem__(k, v)

    def unsafe_set(self, k, v):
        super().__setitem__(k, v)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def __getattr__(self, k):
        v = super().get(k)
        if v is None:
            raise AttributeError(self._get_error_string.format(k, tuple(self.keys())))
        return v

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().update(**kwargs)


def convert_nested_dict_to_handler(d, _class=Handler):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = convert_nested_dict_to_handler(v)

    return _class(**d)


def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.items():
        if isinstance(v, dict):
            if k not in d_to_update.keys():
                d_to_update[k] = {}
            update_dict_of_lists(d_to_update[k], **v)
        elif k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]


def bad_values(d):
    failed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v_ = bad_values(v)
            if v_:
                failed[k] = v_
        else:
            if isinstance(v, torch.Tensor):
                v_ = v.item()
            else:
                v_ = v
            if np.isnan(v_) or np.isinf(v_):
                failed[k] = v_

    if len(failed) == 0:
        return False
    return failed


def convert_to_numpy(o):
    if isinstance(o, torch.Tensor):
        o = o.data.cpu().numpy()
        if len(o.shape) == 1 and o.shape[0] == 1:
            o = o[0]
    elif isinstance(o, (torch.cuda.FloatTensor, torch.cuda.LongTensor)):
        o = o.cpu().numpy()
    elif isinstance(o, list):
        for i in range(len(o)):
            o[i] = convert_to_numpy(o[i])
    elif isinstance(o, tuple):
        o_ = tuple()
        for i in range(len(o)):
            o_ = o_ + (convert_to_numpy(o[i]),)
        o = o_
    elif isinstance(o, dict):
        for k in o.keys():
            o[k] = convert_to_numpy(o[k])
    return o


def compute_tsne(X, perplexity=40, n_iter=300, init='pca'):
    from sklearn.manifold import TSNE

    tsne = TSNE(2, perplexity=perplexity, n_iter=n_iter, init=init)
    points = X.tolist()
    return tsne.fit_transform(points)
