'''Simple dense network encoders

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import torch
from torch.nn.parameter import Parameter


logger = logging.getLogger('cortex.models' + __name__)


class myLinear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(myLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight/self.weight.norm(2), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


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


class DenseNet(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_h=64, batch_norm=True, layer_norm=False,
                 dropout=False, nonlinearity='ReLU', n_levels=None):
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
            if layer_norm:
                models.add_module(name + '_ln', LayerNorm(dim_out))
            elif batch_norm:
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

class myDenseNet(nn.Module):
    def __init__(self, dim_in, dim_out=None, dim_h=64, batch_norm=True, layer_norm=False,
                 dropout=False, nonlinearity='ReLU', n_levels=None):
        super(myDenseNet, self).__init__()
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
            models.add_module(name, myLinear(dim_in, dim_out))
            if dropout:
                models.add_module(name + '_do', nn.Dropout1d(p=dropout))
            if layer_norm:
                models.add_module(name + '_ln', LayerNorm(dim_out))
            elif batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm1d(dim_out))
            models.add_module('{}_{}'.format(name, nonlin), nonlinearity)

        if dim_out_:
            name = 'dense_({}/{})_{}'.format(dim_out, dim_out_, 'final')
            models.add_module(name, myLinear(dim_out, dim_out_))

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
