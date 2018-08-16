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
