'''Tests the optimizer functionality.

'''

import copy

import numpy as np

from cortex._lib import optimizer


def test_optimizer(model_with_submodel):
    model = model_with_submodel
    model.build()

    optimizer.setup(model)

    assert set(optimizer.OPTIMIZERS.keys()) == set(['net', 'net2'])

    optimizer.OPTIMIZERS['net'].step()


def test_clipping(model_with_submodel, clip=0.0001):
    model = model_with_submodel
    model.build()

    optimizer.setup(model, clipping=clip)

    model.train_step()

    optimizer.OPTIMIZERS['net'].step()

    params = model.nets.net.parameters()

    for p in params:
        print(p.min(), p.max())
        assert p.max() <= clip
        assert -p.min() <= clip


def test_gradient(model_with_submodel):
    model = model_with_submodel
    model.build()

    optimizer.setup(model, learning_rate=1.0, optimizer='SGD')

    model.routine(auto_input=True)

    net_loss = model.losses['net']
    parameters = copy.deepcopy(list(model.nets.net.parameters()))
    print(parameters[0][0])
    print(net_loss)

    print('Stepping')
    net_loss.backward()
    grad = [p.grad for p in model.nets.net.parameters()]

    print('grad', grad[0][0])
    model._optimizers['net'].step()

    grad = [p.grad for p in model.nets.net.parameters()]

    for p1, p2, g in zip(list(model.nets.net.parameters()), parameters, grad):
        print(p1[0], p2[0], g[0])
        print('diff', p1[0] - p2[0])
        p1 = p1.data.numpy()
        p2 = p2.data.numpy()
        g = g.data.numpy()
        assert np.allclose(p1, p2 - g)
