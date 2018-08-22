'''Residual encoder / decoder

'''
import logging

import torch.nn as nn

from .base_network import BaseNet
from .modules import View
from .utils import (apply_nonlinearity,
                    finish_layer_2d, get_nonlinearity)
from .SpectralNormLayer import SNConv2d, SNLinear


logger = logging.getLogger('cortex.models' + __name__)


class ConvMeanPool(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None,
                 prefix='', spectral_norm=False):
        super(ConvMeanPool, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'cmp' + prefix

        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1,
                                       bias=False))
        models.add_module(name + '_pool', nn.AvgPool2d(2,
                                                       count_include_pad=False))
        if nonlinearity:
            models.add_module('{}_{}'.format(
                name, nonlinearity.__class__.__name__), nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class MeanPoolConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix='',
                 spectral_norm=False):
        super(MeanPoolConv, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'mpc' + prefix

        models.add_module(name + '_pool', nn.AvgPool2d(
            2, count_include_pad=False))
        models.add_module(
            name, Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        if nonlinearity:
            models.add_module(
                '{}_{}'.format(name, nonlinearity.__class__.__name__),
                nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix='', spectral_norm=False):
        super(UpsampleConv, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = prefix + '_usc'

        models.add_module(name + '_up', nn.Upsample(scale_factor=2))
        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1,
                                       bias=False))

        if nonlinearity:
            models.add_module(
                '{}_{}'.format(name, nonlinearity.__class__.__name__),
                nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_x, dim_y, f_size, resample=None,
                 name='resblock', nonlinearity='ReLU',
                 spectral_norm=False, **layer_args):
        super(ResBlock, self).__init__()

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        skip_models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)

        if resample not in ('up', 'down'):
            raise Exception('invalid resample value: {}'.format(resample))

        # Skip model
        if resample == 'down':
            conv = MeanPoolConv(dim_in, dim_out, f_size, prefix=name,
                                spectral_norm=spectral_norm)
        else:
            conv = UpsampleConv(dim_in, dim_out, f_size, prefix=name,
                                spectral_norm=spectral_norm)
        skip_models.add_module(name + '_skip', conv)

        finish_layer_2d(models, name, dim_x, dim_y, dim_in,
                        nonlinearity=nonlinearity, **layer_args)

        # Up or down sample
        if resample == 'down':
            conv = Conv2d(dim_in, dim_in, f_size, 1, 1)
            models.add_module(name + '_stage1', conv)
            finish_layer_2d(models, name + '_stage1', dim_x // 2, dim_y // 2,
                            dim_in, nonlinearity=nonlinearity,
                            **layer_args)
        else:
            conv = UpsampleConv(dim_in, dim_out, f_size,
                                prefix=name + '_stage1',
                                spectral_norm=spectral_norm)
            models.add_module(name + '_stage1', conv)
            finish_layer_2d(models, name + '_stage1', dim_x * 2, dim_y * 2,
                            dim_out, nonlinearity=nonlinearity,
                            **layer_args)

        if resample == 'down':
            conv = ConvMeanPool(dim_in, dim_out, f_size, prefix=name,
                                spectral_norm=spectral_norm)
        elif resample == 'up':
            conv = Conv2d(dim_out, dim_out, f_size, 1, 1)
        else:
            raise Exception('invalid resample value')

        models.add_module(name + '_stage2', conv)

        self.models = models
        self.skip_models = skip_models

    def forward(self, x):
        x_ = x
        x = self.models(x_)
        x_ = self.skip_models(x_)
        return x + x_


class ResDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, f_size=3, dim_h=64, n_steps=3,
                 nonlinearity='ReLU', output_nonlinearity=None, **layer_args):
        super(ResDecoder, self).__init__()
        models = nn.Sequential()
        dim_h_ = dim_h

        logger.debug('Output shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_h = dim_h_
        nonlinearity = get_nonlinearity(nonlinearity)
        self.output_nonlinearity = output_nonlinearity

        for n in range(n_steps):
            dim_x //= 2
            dim_y //= 2
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_h

        name = 'initial_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module(name + '_reshape', View(-1, dim_h, dim_x, dim_y))
        finish_layer_2d(models, name, dim_x, dim_y, dim_h,
                        nonlinearity=nonlinearity, **layer_args)
        dim_out = dim_h

        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in // 2
            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size,
                                resample='up', name=name, **layer_args)
            models.add_module(name, resblock)
            dim_x *= 2
            dim_y *= 2

        name = 'conv_({}/{})_{}'.format(dim_in, dim_out, 'final')
        finish_layer_2d(models, name, dim_x, dim_y, dim_out,
                        nonlinearity=nonlinearity, **layer_args)
        models.add_module(name, nn.ConvTranspose2d(
            dim_out, dim_out_, f_size, 1, 1, bias=False))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        x = self.models(x)

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)


class ResEncoder(BaseNet):
    def __init__(self, shape, dim_out=None, dim_h=64,
                 fully_connected_layers=None,
                 f_size=3, n_steps=3, nonlinearity='ReLU',
                 output_nonlinearity=None, spectral_norm=False, **layer_args):
        super(ResEncoder, self).__init__(
            nonlinearity=nonlinearity, output_nonlinearity=output_nonlinearity)

        Conv2d = SNConv2d if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []
        if isinstance(fully_connected_layers, int):
            fully_connected_layers = [fully_connected_layers]

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape
        dim_out = dim_h

        name = 'conv_({}/{})_0'.format(dim_in, dim_out)
        self.models.add_module(name, Conv2d(dim_in, dim_out,
                                            f_size, 1, 1,
                                            bias=False))

        dim_out = dim_h
        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in * 2

            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size,
                                resample='down', name=name,
                                spectral_norm=spectral_norm, **layer_args)
            self.models.add_module(name, resblock)

            dim_x //= 2
            dim_y //= 2

        final_depth = dim_out
        dim_out = dim_x * dim_y * dim_out
        self.models.add_module('final_reshape', View(-1, dim_out))

        self.models.add_module('final_reshape_{}x{}x{}to{}'
                               .format(dim_x, dim_y, final_depth, dim_out),
                               View(-1, dim_out))

        dim_out = self.add_linear_layers(dim_out, fully_connected_layers,
                                         Linear=Linear, **layer_args)
        self.add_output_layer(dim_out, dim_out_, Linear=Linear)
