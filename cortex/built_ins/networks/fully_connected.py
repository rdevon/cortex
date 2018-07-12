'''Simple dense network encoders

'''

import logging

import torch.nn as nn

from .utils import apply_nonlinearity, finish_layer_1d, get_nonlinearity


logger = logging.getLogger('cortex.arch' + __name__)


class FullyConnectedNet(nn.Module):

    def __init__(self, dim_in, dim_out=None, dim_h=64, nonlinearity='ReLU',
                 n_levels=None, output_nonlinearity=None, **layer_args):

        super(FullyConnectedNet, self).__init__()
        models = nn.Sequential()

        self.output_nonlinearity = output_nonlinearity

        dim_out_ = dim_out

        if isinstance(dim_h, (list, tuple)):
            pass
        elif n_levels:
            dim_h = [dim_h for _ in range(n_levels)]
        else:
            dim_h = [dim_h]

        nonlinearity = get_nonlinearity(nonlinearity)

        dim_out = dim_in
        for i, dim_h in enumerate(dim_h):
            dim_in = dim_out
            dim_out = dim_h
            name = 'dense_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.Linear(dim_in, dim_out))
            finish_layer_1d(models, name, dim_out,
                            nonlinearity=nonlinearity, **layer_args)

        if dim_out_:
            name = 'dense_({}/{})_{}'.format(dim_out, dim_out_, 'final')
            models.add_module(name, nn.Linear(dim_out, dim_out_))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)
        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
