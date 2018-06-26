'''Testing for building models.

'''

import numpy as np
import pytest
import torch
import torch.optim as optim

from cortex._lib.models import MODEL_PLUGINS
from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin, register_model


arg1 = 'a'
arg2 = 'b'
arg1_help = 'Input dimension'
arg2_help = 'Output dimension'


@pytest.fixture
def cls():
    _build_doc = '''

    Args:
        {arg1}: {arg1_help}
        {arg2}: {arg2_help}

    '''.format(arg1=arg1, arg1_help=arg1_help, arg2=arg2, arg2_help=arg2_help)

    class TestModel(ModelPlugin):
        def build(self, a=17, b=19):
            self.nets.net = FullyConnectedNet(a, b)
        build.__doc__ = _build_doc

        def routine(self, A):
            net = self.nets.net
            output = net(A)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

    return TestModel


class DummyData():
    def __init__(self, dim):
        d_ = np.arange(50 * dim)
        d_ = d_.reshape((50, dim)).astype('float32') * 0.01

        self._data = torch.tensor(d_)
        self.i = 0
        self.bs = 5

    def next(self):
        if self.i >= 10:
            self.i = 0
            raise StopIteration

        d = self._data[self.i * self.bs: (self.i + 1) * self.bs]
        return d

    def __getitem__(self, item):
        return self._data


def test_class(cls):
    assert cls._help[arg1] == arg1_help, cls.help[arg1]
    assert cls._help[arg2] == arg2_help, cls.help[arg2]
    assert cls._kwargs[arg1] == 17, cls.kwargs[arg1]
    assert cls._kwargs[arg2] == 19, cls.kwargs[arg2]


def test_subplugin(cls):
    class TestModel2(ModelPlugin):
        def __init__(self, contract):
            super().__init__()
            self.submodel = cls(contract=contract)

        def build(self, d=31, c=23):
            kwargs = self.submodel.get_kwargs(self.submodel.build)
            self.submodel.build(**kwargs)
            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, B):
            inputs = self.submodel.get_inputs(self.submodel.build)
            kwargs = self.submodel.get_kwargs(self.submodel.build)
            self.subplugin.run(*inputs, **kwargs)

            net = self.nets.net
            output = net(B)

            self.losses.net = output.sum()
            self.losses.net2 = self.subplugin.losses.net
            self.results.output = output.sum().item()
            self.results.output2 = self.subplugin.results.output

        def eval(self):
            pass

        def procedure(self, n_steps=1):
            self.data.next()

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net')
    )

    model = TestModel2(contract)

    kwargs = model.get_kwargs(model.build)
    try:
        model.build(**kwargs)
        assert 0
    except KeyError:
        pass

    ModelPlugin._all_nets.clear()

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net2')
    )
    model = TestModel2(contract)

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

    model.eval(*inputs, **kwargs)

    params = list(model.nets.net.parameters())

    op = optim.SGD(params, lr=0.0001)

    model._optimizers = dict(net=op)

    model.routine(*inputs, **kwargs)

    model.train_step()
    model.train_step()
    model.train_step()

    print(model._all_epoch_results)
    print(model._all_epoch_losses)
    print(model._all_epoch_times)
