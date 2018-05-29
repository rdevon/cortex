"""
Utils for nets
"""

import logging
from torch import nn
LOGGER = logging.getLogger('cortex.arch.modules' + __name__)


def get_nonlinearity(nonlinearity=None):
    """
    TODO
    :param nonlinearity:
    :type nonlinearity:
    :return:
    :rtype:
    """
    if not nonlinearity:
        pass
    elif callable(nonlinearity):
        if nonlinearity == nn.LeakyReLU:
            nonlinearity = nonlinearity(0.02, inplace=True)
        else:
            nonlinearity = nonlinearity()
    elif hasattr(nn, nonlinearity):
        nonlinearity = getattr(nn, nonlinearity)
        if nonlinearity == 'LeakyReLU':
            nonlinearity = nonlinearity(0.02, inplace=True)
        else:
            nonlinearity = nonlinearity()
    else:
        raise ValueError(nonlinearity)

    return nonlinearity


def finish_layer_2d(models, name, dim_x, dim_y, dim_out,
                    dropout=False, layer_norm=False, batch_norm=False, nonlinearity=None):
    """
    TODO
    :param models:
    :type models:
    :param name:
    :type name:
    :param dim_x:
    :type dim_x:
    :param dim_y:
    :type dim_y:
    :param dim_out:
    :type dim_out:
    :param dropout:
    :type dropout:
    :param layer_norm:
    :type layer_norm:
    :param batch_norm:
    :type batch_norm:
    :param nonlinearity:
    :type nonlinearity:
    """
    if layer_norm and batch_norm:
        LOGGER.warning('Ignoring layer_norm because batch_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout2d(p=dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm((dim_out, dim_x, dim_y)))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))

    if nonlinearity:
        models.add_module('{}_{}'.format(name, nonlinearity.__class__.__name__), nonlinearity)


def finish_layer_1d(models, name, dim_out,
                    dropout=False, layer_norm=False, batch_norm=False, nonlinearity=None):
    """
    TODO
    :param models:
    :type models:
    :param name:
    :type name:
    :param dim_out:
    :type dim_out:
    :param dropout:
    :type dropout:
    :param layer_norm:
    :type layer_norm:
    :param batch_norm:
    :type batch_norm:
    :param nonlinearity:
    :type nonlinearity:
    """
    if layer_norm and batch_norm:
        LOGGER.warning('Ignoring layer_norm because batch_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout(p=dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm(dim_out))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))

    if nonlinearity:
        models.add_module('{}_{}'.format(name, nonlinearity.__class__.__name__), nonlinearity)


def apply_nonlinearity(x, nonlinearity, **nonlinearity_args):
    """
    TODO
    :param x:
    :type x:
    :param nonlinearity:
    :type nonlinearity:
    :param nonlinearity_args:
    :type nonlinearity_args:
    :return:
    :rtype:
    """
    if nonlinearity:
        if callable(nonlinearity):
            x = nonlinearity(x, **nonlinearity_args)
        elif hasattr(nn.functional, nonlinearity):
            x = getattr(nn.functional, nonlinearity)(x, **nonlinearity_args)
        else:
            raise ValueError()
    return x