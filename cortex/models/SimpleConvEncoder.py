"""
Convoluational encoders
"""

import logging
import torch.nn as nn

from cortex.models import SNConv2D, SNLinear
from cortex.models import View
from .utils import apply_nonlinearity, finish_layer_1d, finish_layer_2d, get_nonlinearity

LOGGER = logging.getLogger('cortex.arch' + __name__)

def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class SimpleConvEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64, fully_connected_layers=None, nonlinearity='ReLU', f_size=4,
                 stride=2, pad=1, min_dim=4, n_steps=None, spectral_norm=False, **layer_args):
        super(SimpleConvEncoder, self).__init__()

        Conv2d = SNConv2D if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        models = nn.Sequential()

        nonlinearity = get_nonlinearity(nonlinearity)

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        LOGGER.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape

        i = 0
        while (dim_x > min_dim and dim_y > min_dim) and (i < n_steps if n_steps else True):
            LOGGER.debug('Input size: {},{}'.format(dim_x, dim_y))
            if i == 0:
                dim_out = dim_h
            else:
                dim_in = dim_out
                dim_out = dim_in * 2
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)
            finish_layer_2d(models, name, dim_x, dim_y, dim_out, nonlinearity=nonlinearity, **layer_args)
            LOGGER.debug('Output size: {},{}'.format(dim_x, dim_y))
            i += 1

        dim_out = dim_x * dim_y * dim_out
        models.add_module('final_reshape', View(-1, dim_out))

        for dim_h in fully_connected_layers:
            dim_in = dim_out
            dim_out = dim_h
            name = 'linear_({}/{})_{}'.format(dim_in, dim_out, 'final')
            models.add_module(name, Linear(dim_in, dim_out))
            finish_layer_1d(models, name, dim_out, nonlinearity=nonlinearity, **layer_args)

        if dim_out_:
            name = 'linear_({}/{})_{}'.format(dim_out, dim_out_, 'out')
            models.add_module(name, Linear(dim_out, dim_out_))

        self.models = models

    def next_size(self, dim_x, dim_y, k, s, p):
        if isinstance(k, int):
            kx, ky = (k, k)
        else:
            kx, ky = k

        if isinstance(s, int):
            sx, sy = (s, s)
        else:
            sx, sy = s

        if isinstance(p, int):
            px, py = (p, p)
        else:
            px, py = p

        return infer_conv_size(dim_x, kx, sx, px), infer_conv_size(dim_y, ky, sy, py)

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)