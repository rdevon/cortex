from cortex._lib import (config, setup_experiment, exp)
from argparse import Namespace
import torch.nn as nn
from cortex.built_ins.models.utils import update_encoder_args
from cortex.built_ins.models.classifier import ImageClassification


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
    args = Namespace(
        classifier_args={'dropout': 0.2},
        classifier_type='convnet',
        criterion=nn.CrossEntropyLoss(),
        clean=False,
        command=None,
        config_file=None,
        device=0,
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        verbosity=1,
        **{
            'data.batch_size': 10,
            'data.copy_to_local': True,
            'data.data_args': None,
            'data.inputs': 'DUMMY',
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
    model = ClassifierDefaults()
    config.set_config()
    model = setup_experiment(args, model=model)
    assert exp.ARGS['data']['batch_size'] != 128
    assert exp.ARGS['data']['batch_size'] == 10
    assert exp.ARGS['data']['inputs'] != 'images'
    assert exp.ARGS['data']['inputs'] == 'DUMMY'


def test_nested_arguments():
    args = Namespace(
        classifier_args={'dropout': 0.2},
        classifier_type='convnet',
        clean=False,
        command=None,
        config_file=None,
        device=0,
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        verbosity=1,
        **{
            'data.batch_size': 10,
            'data.copy_to_local': True,
            'data.data_args': None,
            'data.inputs': None,
            'data.n_workers': 4,
            'data.shuffle': True,
            'data.skip_last_batch': False,
            'data.source': 'CIFAR10',
            'optimizer.clipping': None,
            'optimizer.learning_rate': 0.001,
            'optimizer.model_optimizer_options': None,
            'optimizer.optimizer': 'Sam',
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
    model = ClassifierBCELoss()
    config.set_config()
    model = setup_experiment(args, model=model)


def test_static_override_parameters():
    args = Namespace(
        clean=False,
        command=None,
        config_file=None,
        device=0,
        load_models=None,
        meta=None,
        name=None,
        out_path=None,
        reload=None,
        reloads=None,
        verbosity=1,
        **{
            'data.batch_size': 10,
            'data.copy_to_local': True,
            'data.data_args': None,
            'data.inputs': None,
            'data.n_workers': 4,
            'data.shuffle': True,
            'data.skip_last_batch': False,
            'data.source': 'CIFAR10',
            'optimizer.clipping': None,
            'optimizer.learning_rate': 0.001,
            'optimizer.model_optimizer_options': None,
            'optimizer.optimizer': 'Sam',
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
    model = ClassifierBCELoss()
    config.set_config()
    model = setup_experiment(args, model=model)
    assert isinstance(exp.ARGS['model']['criterion'], nn.BCELoss)
