from collections import OrderedDict
from cortex.built_ins.networks.utils import get_nonlinearity, \
    apply_nonlinearity, finish_layer_1d, finish_layer_2d
from torch import nn
import torch
from cortex.built_ins.networks.modules import View


def test_get_nonlinearity():
    sigmoid = 'sigmoid'
    tanh = 'tanh'
    relu = 'ReLU'
    leakyrelu = 'LeakyReLU'

    relu = get_nonlinearity(relu)
    tanh = get_nonlinearity(tanh)
    leakyrelu = get_nonlinearity(leakyrelu)
    sigmoid = get_nonlinearity(sigmoid)

    assert callable(sigmoid)
    assert callable(tanh)
    assert isinstance(relu, nn.modules.activation.ReLU)
    assert isinstance(leakyrelu, nn.modules.activation.LeakyReLU)


def test_apply_nonlinearity():
    input = torch.Tensor([1., 2., 3.])
    nonlinearity_args = {}
    nonlinear = 'tanh'

    expected_output = torch.nn.functional.tanh(input)
    applied_nonlinearity = apply_nonlinearity(input, nonlinear, **nonlinearity_args)

    assert torch.equal(expected_output, applied_nonlinearity)


def test_finish_layer_1d():
    # Test settings for a GAN
    layer_norm = False
    batch_norm = True
    dropout = False
    nonlinearity = 'ReLU'
    name = 'linear_(64/4096)'
    dim_out = 4096

    model = nn.Sequential(OrderedDict([
        ('linear_(64/4096)', nn.Linear(in_features=64, out_features=4096, bias=True))
    ]))

    finish_layer_1d(model, name, dim_out, dropout, layer_norm, batch_norm, nonlinearity)

    assert isinstance(model[0], torch.nn.modules.linear.Linear)
    assert isinstance(model[1], torch.nn.modules.batchnorm.BatchNorm1d)
    assert isinstance(model[2], torch.nn.modules.activation.ReLU)
    assert model[0].in_features == 64 and model[0].out_features == 4096


def test_finish_layer_2d():
    # Test settings for a GAN
    dim_x = 4
    dim_y = 4
    dim_out = 256
    dropout = False
    layer_norm = False
    batch_norm = True
    nonlinearity = nn.ReLU()
    name = 'reshape'
    batch_norm_1d_layer = nn.BatchNorm1d(4096)
    nonlinear_relu_layer = nn.ReLU()
    view = View()

    model = nn.Sequential(OrderedDict([
        ('linear_(64/4096)', nn.Linear(in_features=64, out_features=4096, bias=True)),
        ('linear_(64/4096)_bn', batch_norm_1d_layer),
        ('linear_(64/4096)_ReLU', nonlinear_relu_layer),
        ('reshape', view)
    ]))

    finish_layer_2d(model, name, dim_x, dim_y, dim_out, dropout, layer_norm, batch_norm, nonlinearity)

    assert isinstance(model[4], torch.nn.modules.batchnorm.BatchNorm2d)
    assert isinstance(model[5], torch.nn.modules.activation.ReLU)
