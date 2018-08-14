from cortex.built_ins.networks.base_network import BaseNet
from torch import nn


def test_base_net():
    base_net = BaseNet()
    assert isinstance(base_net.models, nn.Sequential)
    assert base_net.output_nonlinearity is None
    assert isinstance(base_net.layer_nonlinearity, nn.ReLU)
