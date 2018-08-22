from cortex.built_ins.networks.convnets import SimpleNet, infer_conv_size
from torch import nn
from cortex.built_ins.networks.modules import View
import torch


def test_simple_conv_encoder_init(simple_conv_encoder_image_classification):
    """
    Args:
        simple_conv_encoder_image_classification (@pytest.fixture): SimpleConvEncoder

    Asserts: True is the layers are of the right type and
             the parameters are correct

    """
    # Settings for ImageClassification experiment.

    layers = list(
        simple_conv_encoder_image_classification.models._modules.items())

    # (conv_(3/64)_1): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    assert isinstance(layers[0][1],
                      nn.Conv2d) and layers[0][0] == 'conv_(3/64)_1'
    assert layers[0][1].kernel_size == (4, 4) and layers[0][1].stride == (
        2, 2) and layers[0][1].padding == (1, 1) and layers[0][1].bias is None

    # (conv_(3/64)_1_do): Dropout2d(p=0.2)
    assert isinstance(layers[1][1],
                      nn.Dropout2d) and layers[1][0] == 'conv_(3/64)_1_do'
    assert layers[1][1].p == 0.2

    # (conv_(3 / 64)_1_bn):
    # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    assert isinstance(layers[2][1],
                      nn.BatchNorm2d) and layers[2][0] == 'conv_(3/64)_1_bn'
    assert layers[2][1].eps == 1e-05 and layers[2][1].momentum == 0.1
    assert layers[2][1].affine is True
    assert layers[2][1].track_running_stats is True

    # (conv_(3 / 64)_1_ReLU): ReLU()
    assert isinstance(layers[3][1],
                      nn.ReLU) and layers[3][0] == 'conv_(3/64)_1_ReLU'

    # (conv_(64/128)_2):
    # Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    assert isinstance(layers[4][1],
                      nn.Conv2d) and layers[4][0] == 'conv_(64/128)_2'
    assert layers[4][1].kernel_size == (4, 4) and layers[4][1].stride == (
        2, 2) and layers[4][1].padding == (1, 1) and layers[4][1].bias is None

    # (conv_(64/128)_2_do): Dropout2d(p=0.2)
    assert isinstance(layers[5][1],
                      nn.Dropout2d) and layers[5][0] == 'conv_(64/128)_2_do'
    assert layers[5][1].p == 0.2

    # (conv_(64/128)_2_bn):
    # BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    assert isinstance(layers[6][1],
                      nn.BatchNorm2d) and layers[6][0] == 'conv_(64/128)_2_bn'
    assert layers[6][1].eps == 1e-05 and layers[6][1].momentum == 0.1
    assert layers[6][1].affine is True
    assert layers[6][1].track_running_stats is True

    # (conv_(64 / 128)_2_ReLU): ReLU()
    assert isinstance(layers[7][1],
                      nn.ReLU) and layers[7][0] == 'conv_(64/128)_2_ReLU'

    # (conv_(128/256)_3):
    # Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    assert isinstance(layers[8][1],
                      nn.Conv2d) and layers[8][0] == 'conv_(128/256)_3'
    assert layers[8][1].kernel_size == (4, 4) and layers[8][1].stride == (
        2, 2) and layers[4][1].padding == (1, 1) and layers[8][1].bias is None

    # (conv_(128/256)_3_do): Dropout2d(p=0.2)
    assert isinstance(layers[9][1],
                      nn.Dropout2d) and layers[9][0] == 'conv_(128/256)_3_do'
    assert layers[9][1].p == 0.2

    # (conv_(128 / 256)_3_bn):
    # BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    assert isinstance(layers[10][1],
                      nn.BatchNorm2d) and layers[10][0] == 'conv_(128/256)_3_bn'
    assert layers[10][1].eps == 1e-05 and layers[10][1].momentum == 0.1
    assert layers[10][1].affine is True
    assert layers[10][1].track_running_stats is True

    # (conv_(128/256)_3_ReLU): ReLU()
    assert isinstance(layers[11][1],
                      nn.ReLU) and layers[11][0] == 'conv_(128/256)_3_ReLU'

    # (final_reshape_4x4x256to4096): View()
    assert isinstance(layers[12][1],
                      View) and layers[12][0] == 'final_reshape_4x4x256to4096'

    # (linear_(4096 / 10)_out): Linear(in_features=4096, out_features=10, bias=True)
    assert isinstance(layers[13][1],
                      nn.Linear) and layers[13][0] == 'linear_(4096/10)_out'
    assert layers[13][1].in_features == 4096 and layers[13][1].out_features == 10
    assert layers[13][1].bias is not None


def test_simple_conv_encoder_next_size(
        simple_conv_encoder_image_classification):
    """

    Args:
        simple_conv_encoder_image_classification (@pytest.fixture): SimpleConvEncoder

    Asserts: True if result is a tuple of adequate values.

    """
    dim_x = 32
    dim_y = 32
    k = 4
    s = 2
    p = 1
    output = simple_conv_encoder_image_classification.next_size(
        dim_x, dim_y, k, s, p)
    expected_value = infer_conv_size(dim_x, k, s, p)
    assert isinstance(output, tuple)
    assert output[0] == expected_value and output[1] == expected_value


def test_infer_conv_size():
    """

    Asserts: True if output is result of formula
             (w - k + 2 * p) // s + 1

    """
    w = 32
    k = 4
    s = 2
    p = 1
    output = infer_conv_size(w, k, s, p)
    assert output == (w - k + 2 * p) // s + 1


def test_simple_net_init():
    """

    Asserts:  True if SimpleNet is being initialize with right default
             layers and parameters.

    """
    simple_net = SimpleNet()
    assert isinstance(simple_net.conv1, nn.Conv2d)
    assert simple_net.conv1.in_channels == 1
    assert simple_net.conv1.out_channels == 10
    assert simple_net.conv1.kernel_size == (5, 5)
    assert isinstance(simple_net.conv2, nn.Conv2d)
    assert simple_net.conv2.in_channels == 10
    assert simple_net.conv2.out_channels == 20
    assert simple_net.conv2.kernel_size == (5, 5)
    assert isinstance(simple_net.conv2_drop, nn.Dropout2d)
    assert isinstance(simple_net.fc1, nn.Linear)
    assert simple_net.fc1.in_features == 320
    assert simple_net.fc1.out_features == 50
    assert isinstance(simple_net.fc2, nn.Linear)
    assert simple_net.fc2.in_features == 50
    assert simple_net.fc2.out_features == 10


def test_simple_net_forward():
    """

    Asserts: True if the output's dimension is equal to the input's one
             and that element-wise, the values have changed.

    """
    simple_net = SimpleNet()
    input = torch.randn(128, 1, 32, 32)
    output = simple_net.forward(input)
    equivalent = torch.equal(input, output)
    assert input.dim() == 4
    assert output.dim() == 2
    assert not equivalent


def test_simple_conv_encoder_forward(simple_conv_encoder_image_classification,
                                     simple_tensor_conv2d):
    """

    Args:
        simple_conv_encoder_image_classification (@pytest.fixture): SimpleConvEncoder
        simple_tensor_conv2d (@pytest.fixture): torch.Tensor

    Asserts: True if the output's dimension is equal to the input's one
             and that element-wise, the values have changed.

    """
    input_dim = simple_tensor_conv2d.dim()
    output = simple_conv_encoder_image_classification.forward(
        simple_tensor_conv2d)
    output_dim = output.dim()
    equivalent = torch.equal(simple_tensor_conv2d, output)
    assert input_dim == 4
    assert output_dim == 2
    assert not equivalent
