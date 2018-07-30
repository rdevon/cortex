from cortex._lib import (config, setup_experiment, exp)
from argparse import Namespace
from cortex.plugins import ModelPlugin
import torch
import torch.nn as nn
import torch.nn.functional as F
from cortex.built_ins.models.utils import update_encoder_args

class ClassifierDefaults(ModelPlugin):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
    )

    def build(self, classifier_type='convnet',
              classifier_args=dict(dropout=0.2)):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''

        classifier_args = classifier_args or {}

        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        Encoder, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.nets.classifier = classifier

    def routine(self, inputs, targets, criterion=nn.BCELoss()):
        '''

        Args:
            criterion: Classifier criterion.

        '''

        classifier = self.nets.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        loss = criterion(outputs, targets)
        correct = 100. * predicted.eq(
            targets.data).cpu().sum() / targets.size(0)

        self.losses.classifier = loss
        self.results.accuracy = correct

    def predict(self, inputs):
        classifier = self.nets.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        return predicted

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)
        self.add_image(images.data, labels=(targets.data, predicted.data),
                       name='gt_pred')

class ClassifierBCELoss(ModelPlugin):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        model=dict(criterion=nn.BCELoss, classifier_args=dict(dim_h=100)),
    )

    def build(self, classifier_type='convnet',
              classifier_args=dict(dropout=0.2)):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''

        classifier_args = classifier_args or {}

        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        Encoder, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.nets.classifier = classifier

    def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            criterion: Classifier criterion.

        '''

        classifier = self.nets.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]
        # We are using the local variable criterion.
        loss = criterion(outputs, targets)
        correct = 100. * predicted.eq(
            targets.data).cpu().sum() / targets.size(0)

        self.losses.classifier = loss
        self.results.accuracy = correct

    def predict(self, inputs):
        classifier = self.nets.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        return predicted

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)
        self.add_image(images.data, labels=(targets.data, predicted.data),
                       name='gt_pred')

#NOTE: exp.ARGS is being populated inside setup_experiment() call
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

# def test_nested_arguments():
#     args = Namespace(
#         classifier_args={'dropout': 0.2},
#         classifier_type='convnet',
#         clean=False,
#         command=None,
#         config_file=None,
#         device=0,
#         load_models=None,
#         meta=None,
#         name=None,
#         out_path=None,
#         reload=None,
#         reloads=None,
#         verbosity=1,
#         **{
#             'data.batch_size': 10,
#             'data.copy_to_local': True,
#             'data.data_args': None,
#             'data.inputs': None,
#             'data.n_workers': 4,
#             'data.shuffle': True,
#             'data.skip_last_batch': False,
#             'data.source': 'CIFAR10',
#             'optimizer.clipping': None,
#             'optimizer.learning_rate': 0.001,
#             'optimizer.model_optimizer_options': None,
#             'optimizer.optimizer': 'Sam',
#             'optimizer.optimizer_options': None,
#             'optimizer.weight_decay': None,
#             'train.archive_every': 10,
#             'train.epochs': 1,
#             'train.eval_during_train': True,
#             'train.eval_only': False,
#             'train.quit_on_bad_values': True,
#             'train.save_on_best': 'losses.classifier',
#             'train.save_on_highest': None,
#             'train.save_on_lowest': None,
#             'train.test_mode': 'test',
#             'train.train_mode': 'train'
#         })
#     model = ClassifierBCELoss()
#     config.set_config()
#     model = setup_experiment(args, model=model)

# def test_static_override_parameters():
#     args = Namespace(
#         clean=False,
#         command=None,
#         config_file=None,
#         device=0,
#         load_models=None,
#         meta=None,
#         name=None,
#         out_path=None,
#         reload=None,
#         reloads=None,
#         verbosity=1,
#         **{
#             'data.batch_size': 10,
#             'data.copy_to_local': True,
#             'data.data_args': None,
#             'data.inputs': None,
#             'data.n_workers': 4,
#             'data.shuffle': True,
#             'data.skip_last_batch': False,
#             'data.source': 'CIFAR10',
#             'optimizer.clipping': None,
#             'optimizer.learning_rate': 0.001,
#             'optimizer.model_optimizer_options': None,
#             'optimizer.optimizer': 'Sam',
#             'optimizer.optimizer_options': None,
#             'optimizer.weight_decay': None,
#             'train.archive_every': 10,
#             'train.epochs': 1,
#             'train.eval_during_train': True,
#             'train.eval_only': False,
#             'train.quit_on_bad_values': True,
#             'train.save_on_best': 'losses.classifier',
#             'train.save_on_highest': None,
#             'train.save_on_lowest': None,
#             'train.test_mode': 'test',
#             'train.train_mode': 'train'
#         })
#     model = ClassifierBCELoss()
#     config.set_config()
#     model = setup_experiment(args, model=model)
#     assert isinstance(exp.ARGS['model']['criterion'], nn.BCELoss)

