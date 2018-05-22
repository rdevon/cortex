'''Utility methods

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

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

        return True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().__init__(**kwargs)

    def __setitem__(self, k, v):
        passes = self.check_key_value(k, v)
        if passes:
            super().__setitem__(k, v)

    def unsafe_set(self, k, v):
        super().__setitem__(k, v)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def __getattr__(self, k):
        if k.startswith('__'):
            return super.get(k)
        try:
            v = super().__getitem__(k)
        except KeyError:
            raise KeyError(self._get_error_string.format(k, tuple(self.keys())))
        return v

    def update(self, **kwargs):
        _kwargs = Handler()
        for k, v in kwargs.items():
            passes = self.check_key_value(k, v)
            if passes:
                _kwargs[k] = v
        super().update(**_kwargs)


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
