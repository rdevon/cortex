'''Simple dense network encoders

'''

import logging

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)



class DenseNet(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_h=64, batch_norm=True, dropout=False, nonlinearity='ReLU', n_levels=None):
        super(DenseNet, self).__init__()
        models = nn.Sequential()

        dim_out_ = dim_out

        if isinstance(dim_h, (list, tuple)):
            pass
        elif n_levels:
            dim_h = [dim_h for _ in xrange(n_levels)]
        else:
            dim_h = [dim_h]

        if hasattr(nn, nonlinearity):
            nonlin = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlin(0.2, inplace=True)
            else:
                nonlinearity = nonlin()
        else:
            raise ValueError(nonlinearity)

        dim_out = dim_in

        for i, dim_h in enumerate(dim_h):
            dim_in = dim_out
            dim_out = dim_h
            name = 'dense_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.Linear(dim_in, dim_out))
            if dropout:
                models.add_module(name + '_do', nn.Dropout1d(p=dropout))
            if batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))
            models.add_module('{}_{}'.format(name, nonlin), nonlinearity)

        if dim_out_:
            name = 'dense_({}/{})_{}'.format(dim_out, dim_out_, 'final')
            models.add_module(name, nn.Linear(dim_out, dim_out_))

        self.models = models

    def forward(self, x, nonlinearity=None, nonlinearity_args=None):
        nonlinearity_args = nonlinearity_args or {}
        x = self.models(x)

        if nonlinearity:
            if callable(nonlinearity):
                x = nonlinearity(x, **nonlinearity_args)
            elif hasattr(F, nonlinearity):
                x = getattr(F, nonlinearity)(x, **nonlinearity_args)
            else:
                raise ValueError()
        return x