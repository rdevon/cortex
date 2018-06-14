from torch import nn
from torchvision import models
from .utils import finish_layer_1d, get_nonlinearity


class AlexNet(models.AlexNet):
    def __init__(self, shape, dim_out=None, fully_connected_layers=None,
                 nonlinearity='ReLU', n_steps=None,
                 **layer_args):
        super(AlexNet, self).__init__()
        fully_connected_layers = fully_connected_layers or []
        self.fc = nn.Sequential()
        dim_out_ = (256 * ((shape[0] + 4 - 10) // 32) *
                    ((shape[1] + 4 - 10) // 32))
        nonlinearity = get_nonlinearity(nonlinearity)
        for dim_h in fully_connected_layers:
            dim_in = dim_out_
            dim_out_ = dim_h
            name = 'linear_%s_%s' % (dim_in, dim_out_)
            self.fc.add_module(name, nn.Linear(dim_in, dim_out_))
            finish_layer_1d(self.fc, name, dim_out_,
                            nonlinearity=nonlinearity, **layer_args)

        if dim_out:
            name = 'dim_out'
            self.fc.add_module(name, nn.Linear(dim_out_, dim_out))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return self.fc(x)
