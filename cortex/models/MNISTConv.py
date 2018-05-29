"""
Convoluational encoders
"""

import logging
import torch.nn as nn
import torch.nn.functional as F

from cortex.models import SNLinear
from cortex.models import SNConv2D
from cortex.models import View
from cortex.models.utils import get_nonlinearity

LOGGER = logging.getLogger('cortex.arch' + __name__)

def infer_conv_size(w, k, s, p):
    x = (w - k + 2 * p) // s + 1
    return x

class MNISTConv(nn.Module):
    def __init__(self, shape, dim_out=1, dim_h=64, batch_norm=True, layer_norm=False, nonlinearity='ReLU',
                 spectral_norm=False):
        super(MNISTConv, self).__init__()
        models = nn.Sequential()

        nonlinearity = get_nonlinearity(nonlinearity)
        Conv2d = SNConv2D if spectral_norm else nn.Conv2d
        Linear = SNLinear if spectral_norm else nn.Linear

        models.add_module('conv1', Conv2d(1, dim_h, 5, 2, 2))
        models.add_module('conv1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv1_ln', nn.LayerNorm(dim_h))
        elif batch_norm:
            models.add_module('conv1_bn', nn.BatchNorm2d(dim_h))

        models.add_module('conv2', Conv2d(dim_h, 2 * dim_h, 5, 2, 2))
        models.add_module('conv2_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('conv2_ln', nn.LayerNorm(2 * dim_h))
        elif batch_norm:
            models.add_module('conv2_bn', nn.BatchNorm2d(2 * dim_h))

        models.add_module('view', View(-1, 2 * dim_h * 7 * 7))

        models.add_module('dense1', Linear(2 * dim_h * 7 * 7, 1024))
        models.add_module('dense1_nonlin', nonlinearity)
        if layer_norm:
            models.add_module('dense1_ln', nn.LayerNorm(1024))
        elif batch_norm:
            models.add_module('dense1_bn', nn.BatchNorm1d(1024))

        models.add_module('dense2', Linear(1024, dim_out))

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

