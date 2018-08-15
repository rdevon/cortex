import pytest


@pytest.fixture
def base_net_model():
    from cortex.built_ins.networks.base_network import BaseNet
    return BaseNet()


@pytest.fixture
def simple_tensor():
    import torch
    return torch.Tensor([1., 2., 3.])


@pytest.fixture
def nonlinearity():
    return dict(
        sigmoid='sigmoid', tanh='tanh', relu='ReLU', leakyrelu='LeakyReLU')
