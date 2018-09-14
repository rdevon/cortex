'''

'''

import numpy as np
import pytest
import torch

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from cortex.plugins import ModelPlugin

import cortex._lib.exp as exp


exp.DEVICE = 'cpu'


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

            d = self._data[self.i * self.bs:(self.i + 1) * self.bs]
            self.i += 1
            return d

        def __getitem__(self, item):
            if item != 'test':
                raise KeyError(item)
            return self._data[self.i * self.bs:(self.i + 1) * self.bs]

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
            self.submodel.build()
            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, B):
            net = self.nets.net
            output = net(B)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

        def train_step(self):
            self.data.next()

            B = self.inputs('B')
            self.routine(B)
            self.optimizer_step()

            A = self.submodel.inputs('A')
            self.submodel.routine(A)
            self.submodel.optimizer_step()

        def eval_step(self):
            self.data.next()

            B = self.inputs('B')
            self.routine(B)

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
            self.submodel1.build()
            self.submodel2.build()

            self.nets.net = FullyConnectedNet(d, c)

        def routine(self, B):
            net = self.nets.net
            output = net(B)

            self.losses.net = output.sum()
            self.results.output = output.sum().item()

        def train_step(self):
            self.data.next()

            B = self.inputs('B')
            self.routine(B)
            self.optimizer_step()

            self.submodel1.train_step()
            self.submodel2.train_step()

        def eval_step(self):
            self.data.next()

            B = self.inputs('B')
            self.routine(B)

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
        kwargs=dict(a='d'), nets=dict(net='net2'), inputs=dict(A='test'))

    model = model_class_with_submodel(
        sub_contract=sub_contract, contract=contract)
    model._data = data
    model.submodel._data = data

    model.kwargs.update(**kwargs)

    return model


@pytest.fixture
def args():
    from argparse import Namespace
    import torch.nn as nn
    return Namespace(
        classifier_args={'dropout': 0.2},
        classifier_type='convnet',
        clean=False,
        command=None,
        config_file=None,
        criterion=nn.CrossEntropyLoss(),
        device='cpu',
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        autoreload=False,
        verbosity=1,
        load_networks=False,
        **{
            'data.batch_size': 128,
            'data.copy_to_local': True,
            'data.data_args': None,
            'data.inputs': {
                'inputs': 'images'
            },
            'data.n_workers': 4,
            'data.shuffle': True,
            'data.skip_last_batch': False,
            'data.source': 'CIFAR10',
            'optimizer.clipping': None,
            'optimizer.learning_rate': 0.001,
            'optimizer.model_optimizer_options': None,
            'optimizer.optimizer': 'Adam',
            'optimizer.optimizer_options': None,
            'optimizer.weight_decay': None,
            'train.archive_every': 10,
            'train.epochs': 1,
            'train.eval_during_train': True,
            'train.eval_only': False,
            'train.quit_on_bad_values': True,
            'train.save_on_best': 'losses.classifier',
            'train.save_on_highest': None,
            'train.save_on_lowest': None,
            'train.test_mode': 'test',
            'train.train_mode': 'train'
        })


@pytest.fixture
def classifier_modified():
    from cortex.built_ins.models.utils import update_encoder_args
    from cortex.built_ins.models.classifier import ImageClassification

    class ClassifierModified(ImageClassification):
        defaults = dict(
            data=dict(batch_size=128, inputs=dict(inputs='images')),
            optimizer=dict(optimizer='Adam', learning_rate=1e-3),
            train=dict(epochs=200, save_on_best='losses.classifier'),
            model=dict(classifier_type='resnet'))

        def build(self,
                  classifier_type='convnet',
                  classifier_args=dict(dropout=0.2)):
            classifier_args = classifier_args or {}
            shape = self.get_dims('x', 'y', 'c')
            dim_l = self.get_dims('labels')
            Encoder, args = update_encoder_args(
                shape, model_type=classifier_type, encoder_args=classifier_args)
            args.update(**classifier_args)
            classifier = Encoder(shape, dim_out=dim_l, **args)
            self.nets.classifier = classifier

    return ClassifierModified()


@pytest.fixture
def base_net_model():
    from cortex.built_ins.networks.base_network import BaseNet
    return BaseNet()


@pytest.fixture
def simple_tensor():
    import torch
    return torch.Tensor([1., 2., 3.])


@pytest.fixture
def simple_tensor_conv2d():
    import torch
    return torch.randn(128, 3, 32, 32)


@pytest.fixture
def nonlinearity():
    return dict(
        sigmoid='sigmoid', tanh='tanh', relu='ReLU', leakyrelu='LeakyReLU')


@pytest.fixture
def simple_classifier():
    from cortex.built_ins.models.classifier import SimpleClassifier
    return SimpleClassifier()


@pytest.fixture
def simple_attribute_classifier():
    from cortex.built_ins.models.classifier import SimpleAttributeClassifier
    return SimpleAttributeClassifier()


@pytest.fixture
def image_classification():
    from cortex.built_ins.models.classifier import ImageClassification
    return ImageClassification()


@pytest.fixture
def image_attribute_classification():
    from cortex.built_ins.models.classifier import ImageAttributeClassification
    return ImageAttributeClassification()


@pytest.fixture
def simple_conv_encoder_image_classification():
    from cortex.built_ins.networks.convnets import SimpleConvEncoder
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

    return SimpleConvEncoder(shape, dim_out, dim_h, fully_connected_layers,
                             nonlinearity, output_nonlinearity, f_size, stride,
                             pad, min_dim, n_steps, spectral_norm, **layer_args)
