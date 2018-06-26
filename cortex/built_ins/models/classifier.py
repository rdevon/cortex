'''Simple classifier model

'''


from cortex.plugins import (register_plugin, ModelPlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from .utils import update_encoder_args


class SimpleClassifierBuild(ModelPlugin):
    '''Build a simple feed-forward classifier.

    '''
    plugin_name = 'simple_classifier'
    plugin_nets = ['simple_classifier']

    def build(self, dim_in: int=None, dim_h=[200, 200], classifier_args={}):
        '''

        Args:
            dim_in (int): Input size
            dim_out (int): Output size
            dim_h (:obj:`list` of :obj:`int`): Hidden layer sizes
            classifier_args: Extra arguments for building the classifier

        '''
        dim_l = self.get_dims('labels')
        classifier = FullyConnectedNet(dim_in, dim_h=dim_h,
                                       dim_out=dim_l, **classifier_args)
        self.nets.simple_classifier = classifier

    def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            criterion: Classifier criterion.

        '''

        classifier = self.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        loss = criterion(outputs, targets)
        correct = 100. * predicted.eq(
            targets.data).cpu().sum() / targets.size(0)

        self.losses.classifier = loss
        self.results['accuracy'] = correct

    def predict(self, inputs):
        classifier = self.classifier

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        return predicted

    def visualize(self, images, inputs, targets):
        predicted = self.predict(inputs)

        self.add_image(inputs.data, labels=(targets.data, predicted.data),
                       name='gt_pred')


class ImageClassification(SimpleClassifierBuild):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''

    data_defaults = dict(batch_size=128)
    optimizer_defaults = dict(optimizer='Adam', learning_rate=1e-4)
    train_defaults = dict(epochs=200, save_on_best='losses.classifier')

    def build(self, images, classifier_type='convnet',
              classifier_args=dict(dropout=0.2)):
        '''Builds a simple image classifier.

        Args:
            images: Tensor of images.
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
            batch_norm, layer_norm, etc.

        '''
        classifier_args = classifier_args or {}

        shape = images.size(1, 2, 3)
        dim_l = self.get_dims('labels')

        Encoder, args = update_encoder_args(
            shape, model_type=classifier_type, encoder_args=classifier_args)

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.nets.classifier = classifier


register_plugin(ImageClassification)
