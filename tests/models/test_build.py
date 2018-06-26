'''Testing for building models.

'''

import torch.optim as optim

from cortex._lib.models import MODEL_PLUGINS
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin, register_model


def test_class(cls):
    assert cls._help[arg1] == arg1_help, cls.help[arg1]
    assert cls._help[arg2] == arg2_help, cls.help[arg2]
    assert cls._kwargs[arg1] == 17, cls.kwargs[arg1]
    assert cls._kwargs[arg2] == 19, cls.kwargs[arg2]


def test_subplugin(cls2):

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net')
    )

    model = cls2(sub_contract=contract)

    kwargs = model.get_kwargs(model.build)
    try:
        model.build(**kwargs)
        assert 0
    except KeyError:
        pass

    ModelPlugin._all_nets.clear()

    contract = dict(
        kwargs=dict(a='c'),
        nets=dict(net='net2')
    )
    model = cls2(contract)

    kwargs = model.get_kwargs(model.build)
    model.build(**kwargs)
    print(model.nets['net'].parameters())


def test_register(cls):
    register_model(cls)
    assert cls in MODEL_PLUGINS.values()


def test_build(cls):
    ModelPlugin._all_nets.clear()
    kwargs = {arg1: 11, arg2: 13}

    model = cls()
    model.kwargs.update(**kwargs)

    model.build()

    print(model.nets)
    assert isinstance(model.nets.net, FullyConnectedNet)


def test_routine(cls):
    ModelPlugin._all_nets.clear()

    kwargs = {arg1: 11, arg2: 13}
    data = DummyData(11)

    contract = dict(inputs=dict(A='test'))
    model = cls(contract=contract)
    model._data = data
    model.kwargs.update(**kwargs)

    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)

    inputs = model.get_inputs(model.routine)
    kwargs = model.get_kwargs(model.routine)

    try:
        model.routine(*inputs, **kwargs)
        assert 0  # optimizer hasn't been set
    except KeyError:
        pass

    assert 'net' in list(model._training_nets.values())[0], model._training_nets

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)
    model._optimizers = dict(net=op)

    model.routine(*inputs, **kwargs)

    model._reset_epoch()

    model.train_step()
    model.train_step()
    model.train_step()

    print(model._all_epoch_results)
    print(model._all_epoch_losses)
    print(model._all_epoch_times)

    assert len(list(model._all_epoch_results.values())[0]) == 3
    assert len(list(model._all_epoch_losses.values())[0]) == 3
    assert len(list(model._all_epoch_times.values())[0]) == 3


def test_routine_with_submodels(cls2):
    ModelPlugin._all_nets.clear()

    kwargs = {'d': 11, 'c': 13}
    data = DummyData(11)

    contract = dict(inputs=dict(B='test'))
    sub_contract = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net2'),
        inputs=dict(A='test')
    )

    model = cls2(sub_contract=sub_contract, contract=contract)
    model._data = data
    model.submodel._data = data
    model.kwargs.update(**kwargs)

    inputs = model.get_inputs(model.build)
    kwargs = model.get_kwargs(model.build)
    model.build(*inputs, **kwargs)

    params = list(model.nets.net.parameters())
    op = optim.SGD(params, lr=0.0001)

    params2 = list(model.nets.net2.parameters())
    op2 = optim.SGD(params2, lr=0.001)

    model._optimizers = dict(net=op, net2=op2)
    model.submodel._optimizers = dict(net=op, net2=op2)

    model.train_step()
    model.train_step()
    model.train_step()