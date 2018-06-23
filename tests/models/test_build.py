'''Testing for building models.

'''

import pytest

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

        def routine(self, inputs):
            net = self.nets.net
            output = net(inputs)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

    return TestModel


def test_class(cls):
    assert cls._help[arg1] == arg1_help, cls.help[arg1]
    assert cls._help[arg2] == arg2_help, cls.help[arg2]
    assert cls._kwargs[arg1] == 17, cls.kwargs[arg1]
    assert cls._kwargs[arg2] == 19, cls.kwargs[arg2]
    assert cls._args[0] == 'inputs'


def test_subplugin(cls):
    class TestModel2(ModelPlugin):
        def __init__(self, contract):
            super().__init__()
            self.submodel = cls(contract=contract)

        def build(self, d=31, c=23):
            self.submodel.build(**self.kwargs)
            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, inputs):
            self.subplugin.run(inputs, **self.kwargs)

            net = self.nets.net
            output = net(inputs)

            self.losses.net = output.sum()
            self.losses.net2 = self.subplugin.losses.net
            self.results.output = output.sum().item()
            self.results.output2 = self.subplugin.results.output

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net')
    )

    model = TestModel2(contract)

    try:
        model.build(**model.kwargs)
        assert 0
    except KeyError:
        pass

    ModelPlugin._nets.clear()

    contract = dict(
        kwargs=dict(b='c'),
        nets=dict(net='net2')
    )
    model = TestModel2(contract)

    print(model.kwargs)
    model.build(**model.kwargs)
    print(model.nets['net'].parameters())
    #print(model._help)


def test_register(cls):
    register_model(cls)

    model = cls()
    #print(model.nets)
    #assert isinstance(model, ModelPlugin)

    #print(model.builds)
    #print(model.routines)

def test_kwargs():
    kwargs = {arg1: 11, arg2: 13}


def test_build():
    '''
    model.set_kwargs(**kwargs)

    model.build(model.kwargs.build)

    print(model.nets.net)
    assert isinstance(model.nets.net, FullyConnectedNet)
    '''