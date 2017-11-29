'''Convoluational encoders

'''

import logging

import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)


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


class SimpleConvEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64, batch_norm=True, dropout=False, nonlinearity='ReLU',
                 f_size=4, stride=2, pad=1, min_dim=4):
        super(SimpleConvEncoder, self).__init__()
        models = nn.Sequential()

        dim_out_ = dim_out

        if hasattr(nn, nonlinearity):
            nonlin = getattr(nn, nonlinearity)
            if nonlinearity == 'LeakyReLU':
                nonlinearity = nonlin(0.2, inplace=True)
            else:
                nonlinearity = nonlin()
        else:
            raise ValueError(nonlinearity)

        logger.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape
        dim_out = dim_h

        name = 'conv_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
        models.add_module('{}_{}'.format(name, nonlin), nonlinearity)
        if dropout:
            models.add_module(name + '_do', nn.Dropout2d(p=dropout))
        dim_x //= 2
        dim_y //= 2

        i = 0
        while dim_x > min_dim and dim_y > min_dim:
            dim_in = dim_out
            dim_out = dim_in * 2
            name = 'conv_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            models.add_module(name, nn.Conv2d(dim_in, dim_out, f_size, stride, pad, bias=False))
            if dropout:
                models.add_module(name + '_do', nn.Dropout2d(p=dropout))
            if batch_norm:
                models.add_module(name + '_bn', nn.BatchNorm2d(dim_out))
            models.add_module('{}_{}'.format(name, nonlin), nonlinearity)
            dim_x //= 2
            dim_y //= 2
            i += 1

        if dim_out_:
            name = 'conv_({}/{})_{}'.format(dim_out, dim_out_, 'final')
            models.add_module(name, nn.Conv2d(dim_out, dim_out_, f_size, 1, 0, bias=False))

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

'''
def build_basic_conv_encoder(X=None, dim_h=None, use_batch_norm=True,
                             use_dropout=None, n_steps=3, shape=None,
                             nonlinearity=rectify, f_size=4, stride=2, pad=1,
                             name='conv encoder'):
    assert dim_h and shape

    if not use_batch_norm:
        bn = lambda x: x
    else:
        bn = batch_norm

    x = InputLayer(shape=shape, input_var=X, name='x')
    log_shape(logger, x, name, 'input')

    args = dict(stride=stride, pad=pad, nonlinearity=nonlinearity)
    for i in xrange(n_steps):
        if use_dropout: x = dropout(x, p=use_dropout)
        x = bn(Conv2DLayer(x, dim_h * (2 ** i), f_size, **args))
        log_shape(logger, x, name, i)

    log_shape(logger, x, name, 'output')
    return x
'''