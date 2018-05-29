import torch.nn as nn
from cortex.models import SNConv2D
from cortex.models.utils import get_nonlinearity

class ConvMeanPool(nn.Module):
    def __init__(self, dim_in, dim_out, f_size, nonlinearity=None, prefix='', spectral_norm=False):
        super(ConvMeanPool, self).__init__()

        Conv2d = SNConv2D if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)
        name = 'cmp' + prefix

        models.add_module(name, Conv2d(dim_in, dim_out, f_size, 1, 1, bias=False))
        models.add_module(name + '_pool', nn.AvgPool2d(2, count_include_pad=False))
        if nonlinearity:
            models.add_module('{}_{}'.format(name, nonlinearity.__class__.__name__), nonlinearity)

        self.models = models

    def forward(self, x):
        x = self.models(x)
        return x
