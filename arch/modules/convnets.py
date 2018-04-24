'''Convoluational encoders

'''

import logging

import torch.nn as nn
import torch.nn.functional as F

from .modules import View
# from .densenet import nn.LayerNorm


logger = logging.getLogger('cortex.arch' + __name__)


def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    def __init__(self, shape, dim_out=1, dim_h=64, batch_norm=True, layer_norm=False, nonlinearity='ReLU'):
        super(MNISTConv, self).__init__()
        models = nn.Sequential()

        if hasattr(nn, nonlinearity):
            nonlin = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlin(0.02, inplace=True)
            else:
                nonlinearity = nonlin()
        else:
            raise ValueError(nonlinearity)

        models.add_module('conv1', nn.Conv2d(1, dim_h, 5, 2, 2))
        models.add_module('conv1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv1_ln', nn.LayerNorm(dim_h))
        elif batch_norm:
            models.add_module('conv1_bn', nn.BatchNorm2d(dim_h))

        models.add_module('conv2', nn.Conv2d(dim_h, 2 * dim_h, 5, 2, 2))
        models.add_module('conv2_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv2_ln', nn.LayerNorm(2 * dim_h))
        elif batch_norm:
            models.add_module('conv2_bn', nn.BatchNorm2d(2 * dim_h))

        models.add_module('view', View(-1, 2 * dim_h * 7 * 7))

        models.add_module('dense1', nn.Linear(2 * dim_h * 7 * 7, 1024))
        models.add_module('dense1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('dense1_ln', nn.LayerNorm(1024))
        elif batch_norm:
            models.add_module('dense1_bn', nn.BatchNorm1d(1024))

        models.add_module('dense2', nn.Linear(1024, dim_out))

        self.models = models

    def forward(self, x, nonlinearity=None, nonlinearity_args=None):
        nonlinearity_args = nonlinearity_args or {}
        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])

        if nonlinearity:
            if callable(nonlinearity):
                x = nonlinearity(x, **nonlinearity_args)
            elif hasattr(F, nonlinearity):
                x = getattr(F, nonlinearity)(x, **nonlinearity_args)
            else:
                raise ValueError()
        return x


class SimpleConvEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64, final_layer=None, batch_norm=True, layer_norm=False, 
                 fully_connected_layers=None, dropout=False, nonlinearity='ReLU', f_size=4, stride=2,
                 pad=1, min_dim=4, n_steps=None):
        super(SimpleConvEncoder, self).__init__()
        models = nn.Sequential()

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []

        if not nonlinearity:
            pass
        elif hasattr(nn, nonlinearity):
            nonlin = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlin(0.02, inplace=True)
            else:
                nonlinearity = nonlin()
        else:
            raise ValueError(nonlinearity)

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape
        #dim_out = dim_h
        '''
        name = 'conv_({}/{})_0'.format(dim_in, dim_out)
        arch.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
        arch.add_module('{}_{}'.format(name, nonlin), nonlinearity)
        if dropout:
            arch.add_module(name + '_do', nn.Dropout2d(p=dropout))
        dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)
        '''

        i = 0
        while (dim_x > min_dim and dim_y > min_dim) and (i < n_steps if n_steps else True):
            logger.debug('Input size: {},{}'.format(dim_x, dim_y))
            if i == 0:
                dim_out = dim_h
            else:
                dim_in = dim_out
                dim_out = dim_in * 2
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
            dim_x, dim_y = self.next_size(dim_x, dim_y, f_size, stride, pad)

            if dropout:
                models.add_module(name + '_do', nn.Dropout2d(p=dropout))
            if layer_norm:
                models.add_module(name + '_ln', nn.LayerNorm((dim_out, dim_x, dim_y)))
            elif batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))
            if nonlinearity:
                # import ipdb; ipdb.set_trace()
                # To avoid the dot in module name
                nonlin_name = str(nonlin).replace(".", "_")
                models.add_module('{}_{}'.format(name, nonlin_name), nonlinearity)
            
            logger.debug('Output size: {},{}'.format(dim_x, dim_y))
            i += 1

        dim_out = dim_x * dim_y * dim_out
        models.add_module('final_reshape', View(-1, dim_out))

        for dim_h in fully_connected_layers:
            dim_in = dim_out
            dim_out = dim_h
            name = 'linear_({}/{})_{}'.format(dim_in, dim_out, 'final')
            models.add_module(name, nn.Linear(dim_in, dim_out))
            if dropout:
                models.add_module(name + '_do', nn.Dropout2d(p=dropout))
            if layer_norm:
                models.add_module(name + '_ln', nn.LayerNorm((dim_out)))
            elif batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))
            if nonlinearity:
                models.add_module('{}_{}'.format(name, nonlin), nonlinearity)

        if final_layer:
            name = 'linear_({}/{})_{}'.format(dim_out, final_layer, 'final')
            models.add_module(name, nn.Linear(dim_out, final_layer))
            models.add_module('{}_{}'.format(name, nonlin), nonlinearity)
            dim_out = final_layer

        if dim_out_:
            name = 'linear_({}/{})_{}'.format(dim_out, dim_out_, 'out')
            models.add_module(name, nn.Linear(dim_out, dim_out_))

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
        nonlinearity_args = nonlinearity_args or {}

        x = self.models(x)
        x = x.view(x.size()[0], x.size()[1])

        if nonlinearity:
            if callable(nonlinearity):
                x = nonlinearity(x, **nonlinearity_args)
            elif hasattr(F, nonlinearity):
                x = getattr(F, nonlinearity)(x, **nonlinearity_args)
            else:
                raise ValueError()
        return x