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
            if isinstance(v, list):
                d_to_update[k] += v
            else:
                d_to_update[k].append(v)
        else:
            if isinstance(v, list):
                d_to_update[k] = v
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


class bcolors:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def bold(s, visdom_mode=False, tb_mode=False):
    if visdom_mode:
        if tb_mode:
            bold_char = ''
            end_char = ''
        else:
            bold_char = '<b>'
            end_char = '</b>'
    else:
        bold_char = bcolors.BOLD
        end_char = bcolors.ENDC
    return bold_char + s + end_char


def underline(s, visdom_mode=False, tb_mode=False):
    if visdom_mode:
        if tb_mode:
            ul_char = ''
            end_char = ''
        else:
            ul_char = '<u>'
            end_char = '</u>'
    else:
        ul_char = bcolors.UNDERLINE
        end_char = bcolors.ENDC
    return  ul_char + s + end_char


def print_hypers(d, prefix=None, s='', visdom_mode=False, level=0, tb_mode=False):
    if visdom_mode:
        newline = '<br>'
        space = '&nbsp;&nbsp;'
    else:
        newline = '\n'
        space = '  '
    prefix = prefix or ''
    for k, v in d.items():
        s += '{}{}{}'.format(newline, space, prefix)
        if level == 0:
            spaces = space * 30
            s += underline('{}: {}'.format(k, spaces), visdom_mode=visdom_mode, tb_mode=tb_mode)
        else:
            s += '{}: '.format(k)

        if isinstance(v, dict) and len(v) > 0:
            s = print_hypers(v, prefix + space, s=s, visdom_mode=visdom_mode, level=level + 1, tb_mode=tb_mode)
        else:
            s += bold('{}'.format(v), visdom_mode=visdom_mode, tb_mode=tb_mode)
    return s