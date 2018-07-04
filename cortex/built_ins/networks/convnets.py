'''Convoluational encoders

'''

import logging

import torch.nn as nn
import torch.nn.functional as F
from .SpectralNormLayer import SNConv2d, SNLinear

from .modules import View
from .utils import (apply_nonlinearity, finish_layer_1d, finish_layer_2d,
                    get_nonlinearity)


logger = logging.getLogger('cortex.arch' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class MNISTConv(nn.Module):
    def __init__(self, shape, dim_out=1, dim_h=64, batch_norm=True,
                 layer_norm=False, nonlinearity='ReLU',
                 output_nonlinearity=None,
                 spectral_norm=False):
        super(MNISTConv, self).__init__()
        models = nn.Sequential()

        self.output_nonlinearity = output_nonlinearity

        nonlinearity = get_nonlinearity(nonlinearity)
        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        models.add_module('conv1', Conv2d(1, dim_h, 5, 2, 2))
        models.add_module('conv1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv1_ln', nn.LayerNorm(dim_h))
        elif batch_norm:
            models.add_module('conv1_bn', nn.BatchNorm2d(dim_h))

        models.add_module('conv2', Conv2d(dim_h, 2 * dim_h, 5, 2, 2))
        models.add_module('conv2_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv2_ln', nn.LayerNorm(2 * dim_h))
        elif batch_norm:
            models.add_module('conv2_bn', nn.BatchNorm2d(2 * dim_h))

        models.add_module('view', View(-1, 2 * dim_h * 7 * 7))

        models.add_module('dense1', Linear(2 * dim_h * 7 * 7, 1024))
        models.add_module('dense1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('dense1_ln', nn.LayerNorm(1024))
        elif batch_norm:
            models.add_module('dense1_bn', nn.BatchNorm1d(1024))

        models.add_module('dense2', Linear(1024, dim_out))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)


class SimpleConvEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64,
                 fully_connected_layers=None, nonlinearity='ReLU',
                 output_nonlinearity=None, f_size=4,
                 stride=2, pad=1, min_dim=4, n_steps=None,
                 spectral_norm=False, **layer_args):
        super(SimpleConvEncoder, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear
        models = nn.Sequential()

        nonlinearity = get_nonlinearity(nonlinearity)
        self.output_nonlinearity = output_nonlinearity

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape

        i = 0
        while ((dim_x > min_dim and dim_y > min_dim) and
               (i < n_steps if n_steps else True)):
            logger.debug('Input size: {},{}'.format(dim_x, dim_y))
            if i == 0:
                dim_out = dim_h
            else:
                dim_in = dim_out
                dim_out = dim_in * 2
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(
                name, Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)
            finish_layer_2d(
                models, name, dim_x, dim_y, dim_out, nonlinearity=nonlinearity,
                **layer_args)
            logger.debug('Output size: {},{}'.format(dim_x, dim_y))
            i += 1

        dim_out = dim_x * dim_y * dim_out
        models.add_module('final_reshape', View(-1, dim_out))

        for dim_h in fully_connected_layers:
            dim_in = dim_out
            dim_out = dim_h
            name = 'linear_({}/{})_{}'.format(dim_in, dim_out, 'final')
            models.add_module(name, Linear(dim_in, dim_out))
            finish_layer_1d(models, name, dim_out, nonlinearity=nonlinearity,
                            **layer_args)

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

        return infer_conv_size(
            dim_x, kx, sx, px), infer_conv_size(dim_y, ky, sy, py)

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
