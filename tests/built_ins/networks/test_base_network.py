from torch import nn


def test_base_net(base_net_model):
    assert isinstance(base_net_model.models, nn.Sequential)
    assert base_net_model.output_nonlinearity is None
    assert isinstance(base_net_model.layer_nonlinearity, nn.ReLU)
