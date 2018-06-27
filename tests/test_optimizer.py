'''Tests the optimizer functionality.

'''

from cortex._lib import optimizer


def test_optimizer(model_with_submodel):
    model = model_with_submodel
    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)

    optimizer.setup(model)

    assert set(optimizer.OPTIMIZERS.keys()) == set(['net', 'net2'])
