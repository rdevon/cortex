'''Simple dense network encoders

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.arch' + __name__)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FullyConnectedNet(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_h=64, batch_norm=True, layer_norm=False,
                 dropout=False, nonlinearity='ReLU', n_levels=None):
        super(FullyConnectedNet, self).__init__()
        models = nn.Sequential()

        dim_out_ = dim_out

        if isinstance(dim_h, (list, tuple)):
            pass
        elif n_levels:
            dim_h = [dim_h for _ in xrange(n_levels)]
        else:
            dim_h = [dim_h]

        if not nonlinearity:
            pass
        elif hasattr(nn, nonlinearity):
            nonlin = nonlinearity
            nonlinearity = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlinearity(0.2, inplace=True)
            else:
                nonlinearity = nonlinearity()
        else:
            raise ValueError(nonlinearity)

        dim_out = dim_in

        for i, dim_h in enumerate(dim_h):
            dim_in = dim_out
            dim_out = dim_h
            name = 'dense_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.Linear(dim_in, dim_out))
            if dropout:
                models.add_module(name + '_do', nn.Dropout(p=dropout))
            if layer_norm:
                models.add_module(name + '_ln', LayerNorm(dim_out))
            elif batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))
            if nonlinearity:
                models.add_module('{}_{}'.format(name, nonlin), nonlinearity)

        if dim_out_:
            name = 'dense_({}/{})_{}'.format(dim_out, dim_out_, 'final')
            models.add_module(name, nn.Linear(dim_out, dim_out_))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
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