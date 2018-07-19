'''Convolutional decoders

'''

import logging

import torch.nn as nn

from .modules import View
from .base_network import BaseNet
from .utils import finish_layer_2d


logger = logging.getLogger('cortex.models' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class SimpleConvDecoder(BaseNet):
    def __init__(self, shape, dim_in=None, initial_layer=None, dim_h=64,
                 nonlinearity='ReLU', output_nonlinearity=None,
                 f_size=4, stride=2, pad=1, n_steps=3, **layer_args):
        super(SimpleConvDecoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        dim_h_ = dim_h
        logger.debug('Input shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_h = dim_h_

        saved_spatial_dimensions = [(dim_x, dim_y)]

        for n in range(n_steps):
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)
            saved_spatial_dimensions.append((dim_x, dim_y))
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_h

        if initial_layer is not None:
            dim_h_ = [initial_layer, dim_out]
        else:
            dim_h_ = [dim_out]

        self.add_linear_layers(dim_in, dim_h_, **layer_args)

        name = 'reshape'
        self.models.add_module(name, View(-1, dim_h, dim_x, dim_y))

        finish_layer_2d(self.models, name, dim_x, dim_y, dim_h,
                        nonlinearity=self.layer_nonlinearity, **layer_args)
        dim_out = dim_h

        for i in range(n_steps):
            dim_in = dim_out

            if i == n_steps - 1:
                pass
            else:
                dim_out = dim_in // 2

            name = 'tconv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            self.models.add_module(
                name, nn.ConvTranspose2d(dim_in, dim_out, f_size, stride, pad,
                                         bias=False))

            finish_layer_2d(self.models, name, dim_x, dim_y, dim_out,
                            nonlinearity=self.layer_nonlinearity, **layer_args)

        self.models.add_module(name + 'f', nn.Conv2d(
            dim_out, dim_out_, 3, 1, 1, bias=False))

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

        return infer_conv_size(
            dim_x, kx, sx, px), infer_conv_size(dim_y, ky, sy, py)
