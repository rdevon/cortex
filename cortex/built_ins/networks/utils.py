'''Utils for networks

'''

import logging

from torch import nn

logger = logging.getLogger('cortex.arch.modules' + __name__)


def get_nonlinearity(nonlinearity=None):
    if not nonlinearity:
        pass
    elif callable(nonlinearity):
        if nonlinearity == nn.LeakyReLU:
            nonlinearity = nonlinearity(0.02, inplace=True)
    elif hasattr(nn, nonlinearity):
        nonlinearity = getattr(nn, nonlinearity)
        if nonlinearity == 'LeakyReLU':
            nonlinearity = nonlinearity(0.02, inplace=True)
        else:
            nonlinearity = nonlinearity()
    elif hasattr(nn.functional, nonlinearity):
        nonlinearity = getattr(nn.functional, nonlinearity)
    else:
        raise ValueError(nonlinearity)
    return nonlinearity


def finish_layer_2d(models, name, dim_x, dim_y, dim_out,
                    dropout=False, layer_norm=False, batch_norm=False,
                    nonlinearity=None):
    if layer_norm and batch_norm:
        logger.warning('Ignoring layer_norm because batch_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout2d(p=dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm((dim_out, dim_x, dim_y)))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))

    if nonlinearity:
        nonlinearity = get_nonlinearity(nonlinearity)
        models.add_module(
            '{}_{}'.format(name, nonlinearity.__class__.__name__),
            nonlinearity)


def finish_layer_1d(models, name, dim_out,
                    dropout=False, layer_norm=False, batch_norm=False,
                    nonlinearity=None):
    if layer_norm and batch_norm:
        logger.warning('Ignoring layer_norm because batch_norm is True')

    if dropout:
        models.add_module(name + '_do', nn.Dropout(p=dropout))

    if layer_norm:
        models.add_module(name + '_ln', nn.LayerNorm(dim_out))
    elif batch_norm:
        models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))

    if nonlinearity:
        nonlinearity = get_nonlinearity(nonlinearity)
        models.add_module(
            '{}_{}'.format(name, nonlinearity.__class__.__name__),
            nonlinearity)


def apply_nonlinearity(x, nonlinearity, **nonlinearity_args):
    if nonlinearity:
        if isinstance(nonlinearity, str):
            nonlinearity = get_nonlinearity(nonlinearity)
        if callable(nonlinearity):
            if isinstance(nonlinearity, nn.PReLU):
                nonlinearity.to(x.device)
            try:
                x = nonlinearity(x, **nonlinearity_args)
            except RuntimeError:
                nonlinearity.to('cpu')
                x = nonlinearity(x, **nonlinearity_args)
        else:
            raise ValueError(nonlinearity, type(nonlinearity))
    return x
