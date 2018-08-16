from torch import nn


def test_base_net(base_net_model):
    """

    Args:
        base_net_model: BaseNet

    Returns: True if BaseNet has an empty nn.Sequential models attribute
             && a default nn.ReLU activation function.

    """
    assert isinstance(base_net_model.models, nn.Sequential)
    assert base_net_model.output_nonlinearity is None
    assert isinstance(base_net_model.layer_nonlinearity, nn.ReLU)


def test_forward_base_net(base_net_model, simple_tensor):

    """

    Args:
        base_net_model: BaseNet
        simple_tensor: torch.Tensor

    Returns: True if the dimension of the output equals the dimension of
             the input.

    """
    base_dimension = simple_tensor.dim()
    output = base_net_model.forward(simple_tensor)
    assert output.dim() == base_dimension

def test_add_linear_layers(base_net_model):
    """

    Args:
        base_net_model: BaseNet

    Returns: True if giving no hidden layers, it returns the dimension
             of the input.

    """
    # Test settings based on ImageClassification.
    dim_in = 4096
    dim_h = []
    dim_ex = None
    Linear = None
    layer_args = dict(batch_norm=True, dropout=0.2)
    output = base_net_model.add_linear_layers(dim_in, dim_h, dim_ex, Linear, **layer_args)
    assert output == dim_in

