from cortex._lib import (config, setup_experiment, exp)
import torch.nn as nn
from cortex.built_ins.models.utils import update_encoder_args
from cortex.built_ins.models.classifier import ImageClassification
from .args_mock import args

class ClassifierDefaults(ImageClassification):
    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
    )
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

class ClassifierBCELoss(ImageClassification):
    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        model=dict(criterion=nn.BCELoss, classifier_args=dict(dim_h=100)),
    )
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


# NOTE: exp.ARGS is being populated inside setup_experiment() call
def test_command_override_static():
    model = ClassifierDefaults()
    config.set_config()
    model = setup_experiment(args, model=model, testmode=True)
    assert exp.ARGS['data']['batch_size'] != 128
    assert exp.ARGS['data']['batch_size'] == 10
    assert exp.ARGS['data']['inputs'] != 'images'
    assert exp.ARGS['data']['inputs'] == 'DUMMY'


def test_nested_arguments():
    model = ClassifierBCELoss()
    config.set_config()
    model = setup_experiment(args, model=model, testmode=True)


# def test_static_override_parameters():
#     model = ClassifierBCELoss()
#     config.set_config()
#     model = setup_experiment(args, model=model)
#     assert isinstance(exp.ARGS['model']['criterion'], nn.BCELoss)
