import torch.nn as nn

from cortex.models import SNConv2D
from cortex.models.utils import get_nonlinearity


class UpsampleConv(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix='', spectral_norm=False):
        super(UpsampleConv, self).__init__()

        Conv2d = SNConv2D if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = prefix + '_usc'

        models.add_module(name + '_up', nn.Upsample(scale_factor=2))
        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))

        if nonlinearity:
            models.add_module('{}_{}'.format(name, nonlinearity.__class__.__name__), nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x
