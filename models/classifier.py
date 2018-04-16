'''Simple classifier model

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)

resnet_args_ = dict(dim_h=64, batch_norm=True, f_size=3, n_steps=4)
mnist_args_ = dict(dim_h=64, batch_norm=True, f_size=5, pad=2, stride=2, min_dim=7, nonlinearity='LeakyReLU')
convnet_args_ = dict(dim_h=64, batch_norm=True, n_steps=3, nonlinearity='LeakyReLU')


DEFAULTS = dict(
    data=dict(batch_size=128),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(dropout=0.2, model_type='convnet'),
    procedures=dict(criterion=nn.CrossEntropyLoss()),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def classify(nets, inputs, criterion=None):
    net = nets['classifier']
    images = inputs['images']
    targets = inputs['targets']
    outputs = net(images, nonlinearity=F.log_softmax)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)
    correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)
    return loss, dict(loss=loss.data[0], accuracy=correct), None, 'accuracy'


def build_model(data_handler, model_type='convnet', dropout=0.2, classifier_args=None):
    classifier_args = classifier_args or {}
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]

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

    net = Encoder(shape, dim_out=dim_l, dropout=dropout, **args)
    return dict(classifier=net), classify