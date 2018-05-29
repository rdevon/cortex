import torch.nn as nn
from cortex.models import View
from cortex.models.utils import apply_nonlinearity


class MNISTDeConv(nn.Module):
    """
    TODO
    """
    def __init__(self, shape, dim_in=64, dim_h=64, batch_norm=True, layer_norm=False):
        super(MNISTDeConv, self).__init__()
        models = nn.Sequential()

        models.add_module('dense1', nn.Linear(dim_in, 1024))
        models.add_module('dense1_relu', nn.ReLU())
        if layer_norm:
            models.add_module('dense1_ln', nn.LayerNorm(1024))
        elif batch_norm:
            models.add_module('dense1_bn', nn.BatchNorm1d(1024))

        models.add_module('dense2', nn.Linear(1024, dim_h * 2 * 7 * 7))
        models.add_module('dense2_relu', nn.ReLU())
        if layer_norm:
            models.add_module('dense2_ln', nn.LayerNorm(2 * dim_h * 7 * 7))
        elif batch_norm:
            models.add_module('dense2_bn', nn.BatchNorm1d(2 * dim_h * 7 * 7))
        models.add_module('view', View(-1, 2 * dim_h, 7, 7))

        models.add_module('deconv1', nn.ConvTranspose2d(2 * dim_h, dim_h, 4, 2, 1))
        models.add_module('deconv1_relu', nn.ReLU())
        if layer_norm:
            models.add_module('deconv1_ln', nn.LayerNorm(dim_h))
        elif batch_norm:
            models.add_module('deconv1_bn', nn.BatchNorm2d(dim_h))

        models.add_module('deconv2', nn.ConvTranspose2d(dim_h, 1, 4, 2, 1))

        self.models = models

    def forward(self, x, nonlinearity=None, **nonlinearity_args):
        """
        TODO
        :param x:
        :type x:
        :param nonlinearity:
        :type nonlinearity:
        :param nonlinearity_args:
        :type nonlinearity_args:
        :return:
        :rtype:
        """
        nonlinearity_args = nonlinearity_args or {}
        x = self.models(x)
        return apply_nonlinearity(x, nonlinearity, **nonlinearity_args)

