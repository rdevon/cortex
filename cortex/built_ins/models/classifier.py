'''Simple classifier model

'''


from cortex.plugins import register_plugin, BuildPlugin, ModelPlugin, RoutinePlugin
import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex.built_ins.networks.fully_connected import FullyConnectedNet
from .utils import update_encoder_args


class ClassifyRoutine(RoutinePlugin):
    '''Routine for doing simple classification.

    '''
    plugin_name = 'classification'
    plugin_nets = ['classifier']
    plugin_inputs = ['inputs', 'targets']
    plugin_optional_inputs = ['images']

    def run(self, criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            criterion: Classifier criterion.

        '''
        classifier = self.nets.classifier
        inputs = self.inputs.inputs
        targets = self.inputs.targets
        images = self.inputs.images

        predicted = self.classify(classifier, inputs, targets, criterion=criterion)

        if images is not None:
            self.visualize(images, targets, predicted)

    def classify(self, classifier, inputs, targets, criterion=None):
        criterion = criterion or nn.CrossEntropyLoss()

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        loss = criterion(outputs, targets)
        correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)

        self.losses.classifier = loss
        self.results[self.name + '_accuracy'] = correct

        return predicted

    def visualize(self, inputs, targets, predicted):
        self.add_image(inputs.data, labels=(targets.data, predicted.data), name=self.name + '_gt_pred')
register_plugin(ClassifyRoutine)


class SimpleClassifierBuild(BuildPlugin):
    '''Build a simple feed-forward classifier.

    '''
    plugin_name = 'simple_classifier'
    plugin_nets = ['simple_classifier']

    def build(self, dim_in: int=None, dim_out: int=None, dim_h=[200, 200], classifier_args={}):
        '''

        Args:
            dim_in (int): Input size
            dim_out (int): Output size
            dim_h (:obj:`list` of :obj:`int`): Hidden layer sizes
            classifier_args: Extra arguments for building the classifier

        '''
        classifier = FullyConnectedNet(dim_in, dim_h=dim_h, dim_out=dim_out, **classifier_args)
        self.add_networks(simple_classifier=classifier)
register_plugin(SimpleClassifierBuild)


class ImageClassifierBuild(BuildPlugin):
    '''Build for a simple image classifier.

    '''
    plugin_name = 'image_classifier'
    plugin_nets = ['image_classifier']

    def build(self, classifier_type='convnet', classifier_args={}):
        '''Builds a simple image classifier.

        Args:
            classifier_type (str): Network type for the classifier.
            classifier_args: Classifier arguments. Can include dropout, batch_norm, layer_norm, etc.

        '''
        classifier_args = classifier_args or {}
        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        Encoder, args = update_encoder_args(shape, model_type=classifier_type, encoder_args=classifier_args)

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
    train_defaults = dict(epochs=200, save_on_best='losses.classifier')

    def __init__(self):
        super().__init__()
        self.add_build(ImageClassifierBuild)
        self.add_routine(ClassifyRoutine, classifier='image_classifier', inputs='data.images', targets='data.targets',
                         images='data.images')
        self.add_train_procedure('classification')
register_plugin(ImageClassification)