import logging
import torch.nn as nn

from cortex.models.ResBlock import ResBlock
from cortex.models.View import View
from .utils import apply_nonlinearity, finish_layer_1d, finish_layer_2d, get_nonlinearity

LOGGER = logging.getLogger('cortex.models' + __name__)

class ResDecoder(nn.Module):
    def __init__(self, shape, dim_in=None, f_size=3, dim_h=64, n_steps=3, nonlinearity='ReLU', **layer_args):
        super(ResDecoder, self).__init__()
        models = nn.Sequential()
        dim_h_ = dim_h

        LOGGER.debug('Output shape: {}'.format(shape))
        dim_x_, dim_y_, dim_out_ = shape

        dim_x = dim_x_
        dim_y = dim_y_
        dim_h = dim_h_
        nonlinearity = get_nonlinearity(nonlinearity)

        for n in range(n_steps):
            dim_x //= 2
            dim_y //= 2
            if n < n_steps - 1:
                dim_h *= 2

        dim_out = dim_x * dim_y * dim_h

        name = 'initial_({}/{})_0'.format(dim_in, dim_out)
        models.add_module(name, nn.Linear(dim_in, dim_out))
        models.add_module(name + '_reshape', View(-1, dim_h, dim_x, dim_y))
        finish_layer_2d(models, name, dim_x, dim_y, dim_h, nonlinearity=nonlinearity, **layer_args)
        dim_out = dim_h

        for i in range(n_steps):
            dim_in = dim_out
            dim_out = dim_in // 2
            name = 'resblock_({}/{})_{}'.format(dim_in, dim_out, i + 1)
            resblock = ResBlock(dim_in, dim_out, dim_x, dim_y, f_size, resample='up', name=name, **layer_args)
            models.add_module(name, resblock)
            dim_x *= 2
            dim_y *= 2

        name = 'conv_({}/{})_{}'.format(dim_in, dim_out, 'final')
        finish_layer_2d(models, name, dim_x, dim_y, dim_out, nonlinearity=nonlinearity, **layer_args)
        models.add_module(name, nn.ConvTranspose2d(dim_out, dim_out_, f_size, 1, 1, bias=False))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        nonlinearity_args = nonlinearity_args or {}
        x = self.models(x)

        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
