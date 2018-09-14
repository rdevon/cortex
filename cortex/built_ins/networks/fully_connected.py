'''Simple dense network encoders

'''

import logging

import torch.nn as nn

from .base_network import BaseNet


logger = logging.getLogger('cortex.arch' + __name__)


class FullyConnectedNet(BaseNet):

    def __init__(self, dim_in, dim_out=None, dim_h=64, dim_ex=None,
                 nonlinearity='ReLU', n_levels=None, output_nonlinearity=None,
                 normalize_input=False, **layer_args):
        super(FullyConnectedNet, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        dim_h = self.get_h(dim_h, n_levels=n_levels)

        if normalize_input:
            self.models.add_module('initial_bn', nn.BatchNorm1d(dim_in))

        dim_in = self.add_linear_layers(dim_in, dim_h, dim_ex=dim_ex,
                                        **layer_args)
        self.add_output_layer(dim_in, dim_out)
