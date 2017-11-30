import logging

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class SimpleConvDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, dim_h=64, batch_norm=True, dropout=False, nonlinearity='ReLU',
                 f_size=4, stride=2, pad=1, n_steps=3):
        super(SimpleConvDecoder, self).__init__()
        models = nn.Sequential()

        dim_h_ = dim_h

        if hasattr(nn, nonlinearity):
            nonlin = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlin(0.2, inplace=True)
            else:
                nonlinearity = nonlin()
        else:
            raise ValueError(nonlinearity)

        logger.debug('Input shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_h = dim_h_

        for n in xrange(n_steps):
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_h

        name = 'initial_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module(name + '_reshape', View(-1, dim_h, dim_x, dim_y))
        if dropout:
            models.add_module(name + '_do', nn.Dropout2d(p=dropout))
        if batch_norm:
            models.add_module(name + '_bn', nn.BatchNorm2d(dim_h))
        dim_out = dim_h

        models.add_module('{}_{}'.format(name, nonlin), nonlinearity)

        for i in xrange(n_steps):
            dim_in = dim_out

            if i == n_steps - 1:
                dim_out = dim_out_
            else:
                dim_out = dim_in // 2
            name = 'tconv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.ConvTranspose2d(dim_in, dim_out, f_size, stride, pad, bias=False))

            if i < n_steps - 1:
                if dropout:
                    models.add_module(name + '_do', nn.Dropout2d(p=dropout))
                if batch_norm:
                    models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))

                models.add_module('{}_{}'.format(name, nonlin), nonlinearity)
            i += 1

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