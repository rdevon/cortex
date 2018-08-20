from cortex.built_ins.networks.convnets import SimpleConvEncoder, SimpleNet
from torch import nn
from cortex.built_ins.networks.modules import View


def test_simple_conv_encoder_init():
    """

    Asserts: True is the layers are of the right type and
            the parameters are correct

    """
    # Settings from ImageClassification experiment.
    shape = [32, 32, 3]
    dim_out = 10
    dim_h = 64
    fully_connected_layers = None
    nonlinearity = 'ReLU'
    output_nonlinearity = None
    f_size = 4
    stride = 2
    pad = 1
    min_dim = 4
    n_steps = 3
    spectral_norm = False
    layer_args = {'batch_norm': True, 'dropout': 0.2}

    simple_conv_encoder = SimpleConvEncoder(
        shape, dim_out, dim_h, fully_connected_layers, nonlinearity,
        output_nonlinearity, f_size, stride, pad, min_dim, n_steps,
        spectral_norm, **layer_args)
    layers = list(simple_conv_encoder.models._modules.items())

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
