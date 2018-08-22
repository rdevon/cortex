from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from torch import nn


def test_fully_connected_build():
    """

    Asserts: True if a the FullyConnectedNet has correct layers and
             attributes.

    """
    dim_in = 4096
    dim_out = 10
    dim_h = 64
    dim_ex = None
    nonlinearity = 'ReLU'
    n_levels = None
    output_nonlinearity = None
    layer_args = {}

    expected_name_linear = 'linear_({}/{})'.format(dim_in, dim_h)
    expected_name_relu = 'linear_({}/{})_{}'.format(dim_in, dim_h, 'ReLU')
    expected_name_out = 'linear_({}/{})_{}'.format(dim_h, dim_out, 'out')

    fully_connected_net = FullyConnectedNet(dim_in, dim_out, dim_h, dim_ex,
                                            nonlinearity, n_levels,
                                            output_nonlinearity, **layer_args)
    layers = list(fully_connected_net.models._modules.items())

    assert layers[0][0] == expected_name_linear
    assert layers[1][0] == expected_name_relu
    assert layers[2][0] == expected_name_out
    assert isinstance(layers[0][1], nn.modules.linear.Linear)
    assert isinstance(layers[1][1], nn.modules.activation.ReLU)
    assert isinstance(layers[2][1], nn.modules.linear.Linear)
    assert layers[0][1].in_features == dim_in
    assert layers[0][1].out_features == dim_h
    assert layers[2][1].in_features == dim_h
    assert layers[2][1].out_features == dim_out
