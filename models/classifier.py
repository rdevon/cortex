'''Simple classifier model

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.convnets import SimpleConvEncoder as Net


logger = logging.getLogger('cortex.models' + __name__)

classifier_args_ = dict(dim_h=256, batch_norm=True, dropout=0.5, nonlinearity='ReLU',
                        f_size=4, stride=2, pad=1, min_dim=4)

DEFAULTS = dict(
    data=dict(batch_size=128),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(),
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


def build_model(data_handler, classifier_args=None):
    classifier_args = classifier_args or {}
    args = classifier_args_
    args.update(**classifier_args)

    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]
    net = Net(shape, dim_out=dim_l, **args)
    return dict(classifier=net), classify