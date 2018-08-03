'''Simple classifier model

'''


from cortex.plugins import (register_plugin, ModelPlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from .utils import update_encoder_args


class SimpleClassifier(ModelPlugin):
    '''Build a simple feed-forward classifier.

    '''

    defaults = dict(
        data=dict(batch_size=128),
        optimizer=dict(optimizer='Adam', learning_rate=1e-4),
        train=dict(epochs=200, save_on_best='losses.classifier'))

    def build(self, dim_in: int=None, classifier_args=dict(dim_h=[200, 200])):

        '''

        Args:
            dim_in (int): Input size
            dim_out (int): Output size
            dim_h (:obj:`list` of :obj:`int`): Hidden layer sizes
            classifier_args: Extra arguments for building the classifier

        '''
        dim_l = self.get_dims('labels')
        classifier = FullyConnectedNet(dim_in, dim_out=dim_l, **classifier_args)
        self.nets.classifier = classifier

    def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss()):
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


class ImageClassification(SimpleClassifier):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    defaults = dict(
        data=dict(batch_size=128, inputs=dict(inputs='images')),
        optimizer=dict(optimizer='Adam', learning_rate=1e-3),
        train=dict(epochs=200, save_on_best='losses.classifier'),
        classifier_args=dict(dropout=0.2))

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


register_plugin(ImageClassification)
