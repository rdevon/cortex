'''Utility methods

'''

import logging
import os

import numpy as np
import torch

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

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
            if isinstance(v, (list, tuple)):
                v_ = []
                for v__ in v:
                    if isinstance(v__, torch.Tensor):
                        v_.append(v__.item())
                    else:
                        v_.append(v__)
                v_ = np.array(v_).sum()
            elif isinstance(v, torch.Tensor):
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
