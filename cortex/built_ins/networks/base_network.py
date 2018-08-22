'''A base network for handling common arguments in cortex models.

This is not necessary to use cortex: these are just convenience networks.

'''

import torch.nn as nn
import torch

from .utils import apply_nonlinearity, get_nonlinearity, finish_layer_1d


class BaseNet(nn.Module):
    '''Basic convenience network for cortex.

    Attributes:
        models: A sequence of

    '''

    def __init__(self, nonlinearity='ReLU', output_nonlinearity=None):
        super(BaseNet, self).__init__()

        self.models = nn.Sequential()

        self.output_nonlinearity = output_nonlinearity
        self.layer_nonlinearity = get_nonlinearity(nonlinearity)

    def forward(self,
                x: torch.Tensor,
                nonlinearity: str = None,
                **nonlinearity_args: dict) -> torch.Tensor:
        self.states = []
        if nonlinearity is None:
            nonlinearity = self.output_nonlinearity
        elif not nonlinearity:
            nonlinearity = None

        for model in self.models:
            x = model(x)
            self.states.append(x)
        x = apply_nonlinearity(x, nonlinearity, **nonlinearity_args)
        return x

    def get_h(self, dim_h, n_levels=None):
        if isinstance(dim_h, (list, tuple)):
            pass
        elif n_levels:
            dim_h = [dim_h for _ in range(n_levels)]
        else:
            dim_h = [dim_h]

        return dim_h

    def add_linear_layers(self,
                          dim_in,
                          dim_h,
                          dim_ex=None,
                          Linear=None,
                          **layer_args):
        Linear = Linear or nn.Linear

        if dim_h is None or len(dim_h) == 0:
            return dim_in

        for dim_out in dim_h:
            name = 'linear_({}/{})'.format(dim_in, dim_out)
            self.models.add_module(name, Linear(dim_in, dim_out))
            finish_layer_1d(
                self.models,
                name,
                dim_out,
                nonlinearity=self.layer_nonlinearity,
                **layer_args)
            dim_in = dim_out
            if dim_ex is not None:
                dim_in += dim_ex

        return dim_out

    def add_output_layer(self, dim_in, dim_out, Linear=None):

        Linear = Linear or nn.Linear
        if dim_out is not None:
            name = 'linear_({}/{})_{}'.format(dim_in, dim_out, 'out')
            self.models.add_module(name, Linear(dim_in, dim_out))


def make_subnet(from_network, n_layers):
    '''Makes a subnet out of another net.

    Shares parameters with original network.

    Args:
        from_network: Network to derive subnet from.
        n_layers: Number of layers from network to use.

    Returns:
        A Subnet for the original network.

    '''
    to_network = BaseNet()
    to_network.models = from_network.models[:n_layers]
    return to_network
