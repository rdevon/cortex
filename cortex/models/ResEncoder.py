import torch.nn as nn
import logging

from cortex.models import SNConv2D, SNLinear
from cortex.models  import ResBlock, View
from cortex.models.utils import get_nonlinearity, finish_layer_1d, apply_nonlinearity

LOGGER = logging.getLogger('cortex.models' + __name__)

class ResEncoder(nn.Module):
    def __init__(self, shape, dim_out=None, dim_h=64, fully_connected_layers=None,
                 f_size=3, n_steps=3, nonlinearity='ReLU', spectral_norm=False, **layer_args):
        super(ResEncoder, self).__init__()

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)

        Conv2d = SNConv2D if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        dim_out_ = dim_out
        fully_connected_layers = fully_connected_layers or []

        LOGGER.debug('Input shape: {}'.format(shape))
        dim_x, dim_y, dim_in = shape
        dim_out = dim_h

        name = 'conv_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        dim_out = dim_h
        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in * 2

            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size, resample='down', name=name,
                                spectral_norm=spectral_norm, **layer_args)
            models.add_module(name, resblock)

            dim_x //= 2
            dim_y //= 2

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

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        x = self.models(x)

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)