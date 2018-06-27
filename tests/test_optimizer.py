'''Tests the optimizer functionality.

'''

from cortex._lib import optimizer


def test_optimizer(model_with_submodel):
    model = model_with_submodel
    model.easy_build()

    optimizer.setup(model)

    assert set(optimizer.OPTIMIZERS.keys()) == set(['net', 'net2'])

    optimizer.OPTIMIZERS['net'].step()


def test_clipping(model_with_submodel, clip=0.0001):
    model = model_with_submodel
    model.easy_build()

    optimizer.setup(model, clipping=clip)

    model.train_step()

    optimizer.OPTIMIZERS['net'].step()

    params = model.nets.net.parameters()

    for p in params:
        print(p.min(), p.max())
        assert p.max() <= clip
        assert -p.min() <= clip
