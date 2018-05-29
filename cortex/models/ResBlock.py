import torch.nn as nn

from cortex.models import SNConv2D
from cortex.models.ConvMeanPool import ConvMeanPool
from cortex.models.MeanPoolConv import MeanPoolConv
from cortex.models.UpSampleConv import UpsampleConv
from cortex.models.utils import get_nonlinearity, finish_layer_2d


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dim_x, dim_y, f_size, resample=None, name='resblock', nonlinearity='ReLU',
                 spectral_norm=False, **layer_args):
        super(ResBlock, self).__init__()

        Conv2d = SNConv2D if spectral_norm else nn.Conv2d

        models = nn.Sequential()
        skip_models = nn.Sequential()
        nonlinearity = get_nonlinearity(nonlinearity)

        if resample not in ('up', 'down'):
            raise Exception('invalid resample value: {}'.format(resample))

        # Skip model
        if resample== 'down':
            conv = MeanPoolConv(dim_in, dim_out, f_size, prefix=name, spectral_norm=spectral_norm)
        else:
            conv = UpsampleConv(dim_in, dim_out, f_size, prefix=name, spectral_norm=spectral_norm)
        skip_models.add_module(name + '_skip', conv)

        finish_layer_2d(models, name, dim_x, dim_y, dim_in, nonlinearity=nonlinearity, **layer_args)

        # Up or down sample
        if resample == 'down':
            conv = Conv2d(dim_in, dim_in, f_size, 1, 1)
            models.add_module(name + '_stage1', conv)
            finish_layer_2d(models, name + '_stage1', dim_x // 2, dim_y // 2, dim_in, nonlinearity=nonlinearity,
                            **layer_args)
        else:
            conv = UpsampleConv(dim_in, dim_out, f_size, prefix=name + '_stage1', spectral_norm=spectral_norm)
            models.add_module(name + '_stage1', conv)
            finish_layer_2d(models, name + '_stage1', dim_x * 2, dim_y * 2, dim_out, nonlinearity=nonlinearity,
                            **layer_args)

        if resample == 'down':
            conv = ConvMeanPool(dim_in, dim_out, f_size, prefix=name, spectral_norm=spectral_norm)
        elif resample == 'up':
            conv = Conv2d(dim_out, dim_out, f_size, 1, 1)
        else:
            raise Exception('invalid resample value')

        models.add_module(name + '_stage2', conv)

        self.models = models
        self.skip_models = skip_models

    def forward(self, x):
        x_ = x
        x = self.models(x_)
        x_ = self.skip_models(x_)
        return x + x_