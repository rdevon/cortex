'''Simple classifier model

'''

import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


resnet_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)
mnist_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
convnet_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU')


def classify(data, models, losses, results, viz, aux_inputs=None, backprop=False, criterion=None):
    classifier = models['classifier']
    inputs, targets = data.get_batch('images', 'targets')
    if aux_inputs is not None:
        inputs_ = inputs
        inputs = aux_inputs
        if not backprop:
            inputs = Variable(inputs.data.cuda(), requires_grad=False)
    else:
        inputs_ = inputs

    outputs = classifier(inputs, nonlinearity=F.log_softmax)

    loss = criterion(outputs, targets)
    predicted = torch.max(outputs.data, 1)[1]
    correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)

    losses.update(classifier=loss)
    results.update(accuracy=correct)
    viz.add_image(inputs_, labels=(targets, predicted), name='gt_pred')


def build_model(data, models, model_type='convnet', dropout=0.2, classifier_args=None):
    classifier_args = classifier_args or {}
    shape = data.get_dims('x', 'y', 'c')
    dim_l = data.get_dims('labels')

    if model_type == 'resnet':
        from .modules.resnets import ResEncoder as Encoder
        args = resnet_args_
    elif model_type == 'convnet':
        from .modules.convnets import SimpleConvEncoder as Encoder
        args = convnet_args_
    elif model_type == 'mnist':
        from .modules.convnets import SimpleConvEncoder as Encoder
        args = mnist_args_
    else:
        raise NotImplementedError(model_type)

    args.update(**classifier_args)

    if shape[0] == 64:
        args['n_steps'] = 4

    classifier = Encoder(shape, dim_out=dim_l, dropout=dropout, **args)
    models.update(classifier=classifier)


ROUTINES = dict(classifier=classify)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=128),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(dropout=0.2, model_type='convnet'),
    routines=dict(criterion=nn.CrossEntropyLoss()),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)