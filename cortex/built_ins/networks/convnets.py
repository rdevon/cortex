'''Convoluational encoders

'''

import logging

import torch.nn as nn
import torch.nn.functional as F
from .SpectralNormLayer import SNConv2d, SNLinear

from .modules import View
from .base_network import BaseNet
from .utils import finish_layer_2d

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


class SimpleConvEncoder(BaseNet):
    def __init__(self, shape, dim_out=None, dim_h=64,
                 fully_connected_layers=None, nonlinearity='ReLU',
                 output_nonlinearity=None, f_size=4,
                 stride=2, pad=1, min_dim=4, n_steps=None, normalize_input=False,
                 spectral_norm=False, last_conv_nonlinearity=True,
                 last_batchnorm=True, **layer_args):
        super(SimpleConvEncoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape

        if isinstance(dim_h, list):
            n_steps = len(dim_h)

        if normalize_input:
            self.models.add_module('initial_bn', nn.BatchNorm2d(dim_in))

        i = 0
        logger.debug('Input size: {},{}'.format(dim_x, dim_y))
        while ((dim_x >= min_dim and dim_y >= min_dim) and
               (i < n_steps if n_steps else True)):
            if i == 0:
                if isinstance(dim_h, list):
                    dim_out = dim_h[0]
                else:
                    dim_out = dim_h
            else:
                dim_in = dim_out
                if isinstance(dim_h, list):
                    dim_out = dim_h[i]
                else:
                    dim_out = dim_in * 2
            conv_args = dict((k, v) for k, v in layer_args.items())
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)

            self.models.add_module(
                name, Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)

            is_last_layer = not((dim_x >= min_dim and dim_y >= min_dim) and
                                (i < n_steps if n_steps else True))

            if is_last_layer:
                if not(last_conv_nonlinearity):
                    nonlinearity = None
                else:
                    nonlinearity = self.layer_nonlinearity

                if not(last_batchnorm):
                    conv_args['batch_norm'] = False

            finish_layer_2d(
                self.models, name, dim_x, dim_y, dim_out,
                nonlinearity=nonlinearity, **conv_args)
            logger.debug('Output size: {},{}'.format(dim_x, dim_y))
            i += 1

        if len(fully_connected_layers) == 0 and dim_out_ is None:
            return

        dim_out__ = dim_out
        dim_out = dim_x * dim_y * dim_out

        self.models.add_module('final_reshape_{}x{}x{}to{}'
                               .format(dim_x, dim_y, dim_out__, dim_out),
                               View(-1, dim_out))

        dim_out = self.add_linear_layers(dim_out, fully_connected_layers,
                                         Linear=Linear, **layer_args)
        self.add_output_layer(dim_out, dim_out_, Linear=Linear)

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
