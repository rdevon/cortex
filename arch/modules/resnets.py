'''Residual encoder / decoder

'''
import logging

import torch.nn as nn
import torch.nn.functional as F

from .densenet import LayerNorm


logger = logging.getLogger('cortex.models' + __name__)


class ConvMeanPool(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix=''):
        super(ConvMeanPool, self).__init__()
        models = nn.Sequential()
        name = 'cmp' + prefix
        models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))
        models.add_module(name + '_pool', nn.AvgPool2d(2, count_include_pad=False))
        if nonlinearity:
            models.add_module(name + '_nonlin', nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class MeanPoolConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix=''):
        super(MeanPoolConv, self).__init__()
        models = nn.Sequential()
        name = 'mpc' + prefix

        models.add_module(name + '_pool', nn.AvgPool2d(2, count_include_pad=False))
        models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        if nonlinearity:
            models.add_module(name + '_nonlin', nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class UpsampleConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix=''):
        super(UpsampleConv, self).__init__()
        models = nn.Sequential()
        name = prefix + '_usc'
        models.add_module(name + '_up', nn.Upsample(scale_factor=2))
        models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        if nonlinearity:
            models.add_module(name + '_nonlin', nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, resample=None, batch_norm=True,
                 layer_norm=False, prefix=''):
        super(ResBlock, self).__init__()
        models = nn.Sequential()
        skip_models = nn.Sequential()
        name = prefix + '_resblock'

        if resample== 'down':
            skip_models.add_module(
                name + '_skip', MeanPoolConv(dim_in, dim_out, f_size, prefix=prefix))
        elif resample == 'up':
            skip_models.add_module(
                name + '_skip', UpsampleConv(dim_in, dim_out, f_size, prefix=prefix))
        else:
            raise Exception('invalid resample value')

        if layer_norm:
            models.add_module(name + '_ln', LayerNorm(dim_in))
        elif batch_norm:
            models.add_module(name + '_bn', nn.BatchNorm2d(dim_in))

        models.add_module('{}_{}'.format(name, 'rectify'), nn.ReLU())

        if resample == 'down':
            models.add_module(name + '_stage1', nn.Conv2d(dim_in, dim_in, f_size, 1, 1))
            if layer_norm:
                models.add_module(name + '_ln2', LayerNorm(dim_in))
            elif batch_norm:
                models.add_module(name + '_bn2', nn.BatchNorm2d(dim_in))
        elif resample == 'up':
            models.add_module(name + '_stage1', UpsampleConv(dim_in, dim_out, f_size, prefix=prefix))
            if layer_norm:
                models.add_module(name + '_ln2', LayerNorm(dim_out))
            elif batch_norm:
                models.add_module(name + '_bn2', nn.BatchNorm2d(dim_out))
        else:
            raise Exception('invalid resample value')

        models.add_module('{}_{}'.format(name, 'rectify2'), nn.ReLU())

        if resample == 'down':
            models.add_module(name + '_stage2', ConvMeanPool(dim_in, dim_out, f_size, prefix=prefix))
        elif resample == 'up':
            models.add_module(name + '_stage2', nn.Conv2d(dim_out, dim_out, f_size, 1, 1))
        else:
            raise Exception('invalid resample value')

        self.models = models
        self.skip_models = skip_models

    def forward(self, x):
        x_ = x
        x = self.models(x_)
        x_ = self.skip_models(x_)
        return x + x_


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ResDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, f_size=3, dim_h=64, batch_norm=True,
                 layer_norm=False, n_steps=3):
        super(ResDecoder, self).__init__()
        models = nn.Sequential()

        dim_h_ = dim_h

        logger.debug('Output shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_h = dim_h_
        nonlinearity = nn.ReLU()

        for n in range(n_steps):
            dim_x //= 2
            dim_y //= 2
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_h

        name = 'initial_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module(name + '_reshape', View(-1, dim_h, dim_x, dim_y))
        if layer_norm:
            models.add_module(name + '_ln', LayerNorm(dim_h))
        elif batch_norm:
            models.add_module(name + '_bn', nn.BatchNorm2d(dim_h))
        dim_out = dim_h

        models.add_module('{}_{}'.format(name, 'relu'), nonlinearity)

        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in // 2
            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, ResBlock(dim_in, dim_out, f_size, resample='up',
                                             batch_norm=batch_norm, layer_norm=layer_norm, prefix=name))

        name = 'conv_({}/{})_{}'.format(dim_in, dim_out, 'final')
        if layer_norm:
            models.add_module(name + '_ln', LayerNorm(dim_out))
        elif batch_norm:
            models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))

        models.add_module('{}_{}'.format(name, 'relu'), nonlinearity)
        models.add_module(name, nn.ConvTranspose2d(dim_out, dim_out_, f_size, 1, 1, bias=False))

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


class ResEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64, f_size=3, batch_norm=True, 
                 layer_norm=False, n_steps=3):
        super(ResEncoder, self).__init__()
        models = nn.Sequential()

        dim_out_ = dim_out

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape
        dim_out = dim_h

        name = 'conv_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        dim_out = dim_h
        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in * 2

            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, ResBlock(dim_in, dim_out, f_size, resample='down',
                                             batch_norm=batch_norm, layer_norm=layer_norm, prefix=name))

            dim_x //= 2
            dim_y //= 2

        if dim_out_:
            models.add_module(name + '_reshape', View(-1, dim_out * dim_x * dim_y))
            name = 'lin_({}/{})_{}'.format(dim_out * dim_x * dim_y, dim_out_, 'final')
            models.add_module(name, nn.Linear(dim_out * dim_x * dim_y, dim_out_))

        self.models = models

    def forward(self, x, nonlinearity=None, nonlinearity_args=None):
        nonlinearity_args = nonlinearity_args or {}
        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])
        if nonlinearity:
            assert False, nonlinearity
            if callable(nonlinearity):
                x = nonlinearity(x, **nonlinearity_args)
            elif hasattr(F, nonlinearity):
                x = getattr(F, nonlinearity)(x, **nonlinearity_args)
            else:
                raise ValueError()
        return x