'''

'''


import numpy as np
import pytest
import torch

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin


@pytest.fixture
def arguments():
    arg1 = 'a'
    arg2 = 'b'
    arg1_help = 'Input dimension'
    arg2_help = 'Output dimension'

    return dict(arg1=arg1, arg2=arg2, arg1_help=arg1_help, arg2_help=arg2_help)


@pytest.fixture
def data_class():
    class DummyData():
        def __init__(self, dim):
            d_ = np.arange(50 * dim)
            d_ = d_.reshape((50, dim)).astype('float32') * 0.01

            self._data = torch.tensor(d_)
            self.i = 0
            self.bs = 5

        def next(self):
            if self.i >= 9:
                self.i = 0
                raise StopIteration

            d = self._data[self.i * self.bs: (self.i + 1) * self.bs]
            self.i +=1
            return d

        def __getitem__(self, item):
            return self._data[self.i * self.bs: (self.i + 1) * self.bs]

        def reset(self, *args, **kwargs):
            self.i = 0

    return DummyData


@pytest.fixture
def model_class(arguments):

    _build_doc = '''

    Args:
        {arg1}: {arg1_help}
        {arg2}: {arg2_help}

    '''.format(**arguments)

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


@pytest.fixture
def model_class_with_submodel(model_class):
    class TestModel2(ModelPlugin):
        def __init__(self, sub_contract=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submodel = model_class(contract=sub_contract)

        def build(self, d=31, c=23):
            kwargs = self.get_kwargs(self.submodel.build)
            self.submodel.build(**kwargs)
            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, B):
            net = self.nets.net
            output = net(B)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

        def train_step(self):
            self.data.next()

            self.easy_routine()
            self.optimizer_step()

            self.submodel.easy_routine()
            self.submodel.optimizer_step()

        def eval_step(self):
            self.data.next()

            kwargs = self.get_kwargs(self.routine)
            inputs = self.get_inputs(self.routine)
            self.routine(*inputs, **kwargs)

            self.submodel.eval_step()

    return TestModel2


@pytest.fixture
def model_class_with_submodel_2(model_class):
    class TestModel3(ModelPlugin):
        def __init__(self, sub_contract1, sub_contract2, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.submodel1 = model_class(contract=sub_contract1)
            self.submodel2 = model_class(contract=sub_contract2)

        def build(self, d=31, c=23):
            kwargs = self.get_kwargs(self.submodel1.build)
            self.submodel1.build(**kwargs)

            kwargs = self.get_kwargs(self.submodel2.build)
            self.submodel2.build(**kwargs)

            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, B):
            net = self.nets.net
            output = net(B)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

        def train_step(self):
            self.data.next()

            inputs = self.get_inputs(self.routine)
            kwargs = self.get_kwargs(self.routine)
            self.routine(*inputs, **kwargs)
            self.optimizer_step()

            self.submodel1.train_step()
            self.submodel2.train_step()

        def eval_step(self):
            self.data.next()

            inputs = self.get_inputs(self.routine)
            kwargs = self.get_kwargs(self.routine)
            self.routine(*inputs, **kwargs)

            self.submodel1.eval_step()
            self.submodel2.eval_step()

    return TestModel3


@pytest.fixture
def model_with_submodel(model_class_with_submodel, data_class):
    ModelPlugin._reset_class()

    kwargs = {'d': 11, 'c': 13}
    data = data_class(11)

    contract = dict(inputs=dict(B='test'))
    sub_contract = dict(
        kwargs=dict(a='d'),
        nets=dict(net='net2'),
        inputs=dict(A='test')
    )

    model = model_class_with_submodel(sub_contract=sub_contract,
                                      contract=contract)
    model._data = data
    model.submodel._data = data

    model.kwargs.update(**kwargs)

    return model
