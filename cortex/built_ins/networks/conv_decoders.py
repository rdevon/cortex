'''Convolutional decoders

'''

import logging

import torch.nn as nn

from .modules import View
from .utils import (apply_nonlinearity, get_nonlinearity, finish_layer_1d,
                    finish_layer_2d)


logger = logging.getLogger('cortex.models' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class MNISTDeConv(nn.Module):
    def __init__(self, shape, dim_in=64, dim_h=64, batch_norm=True, layer_norm=False):
        super(MNISTDeConv, self).__init__()
        models = nn.Sequential()

        models.add_module('dense1', nn.Linear(dim_in, 1024))
        models.add_module('dense1_relu', nn.ReLU())
        if layer_norm:
            models.add_module('dense1_ln', nn.LayerNorm(1024))
        elif batch_norm:
            models.add_module('dense1_bn', nn.BatchNorm1d(1024))

        models.add_module('dense2', nn.Linear(1024, dim_h * 2 * 7 * 7))
        models.add_module('dense2_relu', nn.ReLU())
        if layer_norm:
            models.add_module('dense2_ln', nn.LayerNorm(2 * dim_h * 7 * 7))
        elif batch_norm:
            models.add_module('dense2_bn', nn.BatchNorm1d(2 * dim_h * 7 * 7))
        models.add_module('view', View(-1, 2 * dim_h, 7, 7))

        models.add_module('deconv1', nn.ConvTranspose2d(
            2 * dim_h, dim_h, 4, 2, 1))
        models.add_module('deconv1_relu', nn.ReLU())
        if layer_norm:
            models.add_module('deconv1_ln', nn.LayerNorm(dim_h))
        elif batch_norm:
            models.add_module('deconv1_bn', nn.BatchNorm2d(dim_h))

        models.add_module('deconv2', nn.ConvTranspose2d(dim_h, 1, 4, 2, 1))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)
        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)


class SimpleConvDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, initial_layer=None, dim_h=64,
                 nonlinearity='ReLU', output_nonlinearity=None,
                 f_size=4, stride=2, pad=1, n_steps=3, **layer_args):
        super(SimpleConvDecoder, self).__init__()
        models = nn.Sequential()

        dim_h_ = dim_h

        nonlinearity = get_nonlinearity(nonlinearity)
        self.output_nonlinearity = output_nonlinearity

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
        if initial_layer:
            name = 'initial_({}/{})'.format(dim_in, initial_layer)
            models.add_module(name, nn.Linear(dim_in, initial_layer))
            finish_layer_1d(models, name, initial_layer,
                            nonlinearity=nonlinearity, **layer_args)
            dim_in = initial_layer

        name = 'first_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module(name + '_reshape', View(-1, dim_h, dim_x, dim_y))

        finish_layer_2d(models, name, dim_x, dim_y, dim_h,
                        nonlinearity=nonlinearity, **layer_args)
        dim_out = dim_h

        for i in range(n_steps):
            dim_in = dim_out

            if i == n_steps - 1:
                pass
            else:
                dim_out = dim_in // 2

            name = 'tconv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(
                name, nn.ConvTranspose2d(dim_in, dim_out, f_size, stride, pad,
                                         bias=False))

            finish_layer_2d(models, name, dim_x, dim_y, dim_out,
                            nonlinearity=nonlinearity, **layer_args)

        models.add_module(name + 'f', nn.Conv2d(
            dim_out, dim_out_, 3, 1, 1, bias=False))

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

        return infer_conv_size(
            dim_x, kx, sx, px), infer_conv_size(dim_y, ky, sy, py)

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)
        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
