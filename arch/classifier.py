'''Simple classifier model

'''


from cortex.plugins import BuildPlugin, ExperimentPlugin, RoutinePlugin
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import update_encoder_args


resnet_args_ = dict(dim_h=64, batch_norm=True, dropout=0.2, f_size=3, n_steps=4, fully_connected_layers=[1028])
mnist_args_ = dict(dim_h=64, batch_norm=True, dropout=0.2, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
convnet_args_ = dict(dim_h=64, batch_norm=True, dropout=0.2, n_steps=3, nonlinearity='LeakyReLU')


class ClassifyRoutinePlugin(RoutinePlugin):
    name = 'classification'

    def run(self, key='classifier', criterion=nn.CrossEntropyLoss()):
        '''

        Args:
            criterion: Classifier criterion.

        Returns:

        '''
        classifier = self.models[key]
        inputs, targets = self.get_batch('images', 'targets')

        predicted = self.classify(classifier, inputs, targets, criterion=criterion, key=key)
        self.visualize(inputs, targets, predicted, key=key)

    def classify(self, classifier, inputs, targets, criterion=None, key='classifier'):
        criterion = criterion or nn.CrossEntropyLoss()

        outputs = classifier(inputs)
        predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

        loss = criterion(outputs, targets)
        self.losses[key] = loss

        correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)
        self.results[key + '_accuracy'] = correct

        return predicted

    def visualize(self, inputs, targets, predicted, key='classifier'):
        self.add_image(inputs.data, labels=(targets.data, predicted.data), name=key + '_gt_pred')
ClassifyRoutinePlugin()


class ImageClassifierBuildPlugin(BuildPlugin):
    name = 'image_classifier'
    def build(self, model_type='convnet', classifier_args={}):
        '''Builds a simple image classifier.

        Attributes:
            blah: hello

        Args:
            model_type (str): Model type for the classifier.
            classifier_args: Classifier arguments. Can include dropout, batch_norm, layer_norm, etc.

        Returns:
            whatever

        '''
        classifier_args = classifier_args or {}
        shape = self.get_dims('x', 'y', 'c')
        dim_l = self.get_dims('labels')

        if model_type == 'resnet':
            args = resnet_args_
        elif model_type == 'convnet':
            args = convnet_args_
        elif model_type == 'mnist':
            args = mnist_args_
        elif 'tv' in model_type:
            args = {}
        else:
            raise NotImplementedError(model_type)

        Encoder, args = update_encoder_args(shape, model_type=model_type, encoder_args=args)

        args.update(**classifier_args)

        classifier = Encoder(shape, dim_out=dim_l, **args)
        self.add_models(classifier=classifier)
ImageClassifierBuildPlugin()


DEFAULT_CONFIG = dict(
    data=dict(batch_size=128),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4),
    train=dict(epochs=200, archive_every=10, save_on_best='losses.classifier')
)


class ImageClassificationPlugin(ExperimentPlugin):
    name = 'image_classifier'
    build = ['image_classifier']
    train_routines = ['classification']
    defaults=DEFAULT_CONFIG
ImageClassificationPlugin()

