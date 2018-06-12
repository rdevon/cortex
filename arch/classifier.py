'''Simple classifier model

'''


from cortex.plugins import (register_plugin, BuildPlugin, ModelPlugin,
                            RoutinePlugin)
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import update_encoder_args


class ClassifyRoutine(RoutinePlugin):
    '''Routine for doing simple classification.

    '''
    plugin_name = 'classification'
    plugin_nets = ['classifier']
    plugin_inputs = ['inputs', 'targets']

    def run(self, criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            criterion: Classifier criterion.

        '''
        classifier = self.nets.classifier
        inputs = self.inputs.inputs
        targets = self.inputs.targets

        predicted = self.classify(classifier, inputs, targets,
                                  criterion=criterion)
        self.visualize(inputs, targets, predicted)

    def classify(self, classifier, inputs, targets, criterion=None):
        criterion = criterion or nn.CrossEntropyLoss()

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        loss = criterion(outputs, targets)
        correct = (100. * predicted.eq(targets.data).cpu().sum() /
                   targets.size(0))

        self.losses.classifier = loss
        self.results[self.name + '_accuracy'] = correct

        return predicted

    def visualize(self, inputs, targets, predicted):
        self.add_image(inputs.data, labels=(targets.data, predicted.data),
                       name=self.name + '_gt_pred')
register_plugin(ClassifyRoutine)


class ImageClassifierBuild(BuildPlugin):
    '''Build for a simple image classifier.

    '''
    plugin_name = 'image_classifier'
    plugin_nets = ['image_classifier']

    def build(self, classifier_type='convnet', classifier_args={}):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout,
                             batch_norm, layer_norm, etc.

        '''
        classifier_args = classifier_args or {}
        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        Encoder, args = update_encoder_args(shape, model_type=classifier_type,
                                            encoder_args=classifier_args)

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.add_networks(image_classifier=classifier)
register_plugin(ImageClassifierBuild)


class ImageClassification(ModelPlugin):
    '''Basic image classifier.

    Classifies images using standard convnets.

    '''
    plugin_name = 'image_classifier'

    data_defaults = dict(batch_size=128)
    optimizer_defaults = dict(optimizer='Adam', learning_rate=1e-4)
    train_defaults = dict(epochs=200, archive_every=10,
                          save_on_best='losses.classifier')

    def __init__(self):
        super().__init__()
        self.add_build('image_classifier', image_classifier='my_classifier',
                       name='my_build')
        self.add_routine('classification', classifier='my_classifier',
                         inputs='data.images', targets='data.targets',
                         name='my_classification')
        self.add_train_procedure('my_classification')
register_plugin(ImageClassification)
