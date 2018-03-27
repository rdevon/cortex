'''
Simple Regularized classifier model

example usage:

python main.py reg_classifier -S MNIST -n l1reg -a procedures.regularizer='l1'
python main.py reg_classifier -S MNIST -n l1reg -a procedures.regularizer='l2'
python main.py reg_classifier -S MNIST -n l1reg -a procedures.regularizer='en'

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.convnets import SimpleConvEncoder as Net
from .modules.regularization import Regularizer

logger = logging.getLogger('cortex.models' + __name__)

regularizer_args_ = dict(reg_type='l1', factor=0.01)

classifier_args_ = dict(dim_h=256, batch_norm=True, dropout=0.5, nonlinearity='ReLU',
                        f_size=4, stride=2, pad=1, min_dim=4)

DEFAULTS = dict(
    data=dict(batch_size=128),
    optimizer=dict(
        optimizer='Adam',
        learning_rate=1e-4,
    ),
    model=dict(),
    procedures=dict(criterion=nn.CrossEntropyLoss(), regularizer='l1'),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
    
)


def classify(nets, inputs, criterion=None, regularizer=None, factor=0.0005):
    net = nets['classifier']
    images = inputs['images']
    targets = inputs['targets']
    reg_term = Regularizer(loss=criterion, reg_type=regularizer, factor=factor)
    outputs = net(images, nonlinearity=F.log_softmax)
    loss = reg_term.compute_reg_term(outputs, targets, net.parameters())
    _, predicted = torch.max(outputs.data, 1)
    correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)
    return loss, dict(loss=loss.data[0], accuracy=correct), None, 'accuracy'


def build_model(data_handler, classifier_args=None, regularizer_args=None):
    global REGULARIZER
    classifier_args = classifier_args or {}
    args = classifier_args_
    args.update(**classifier_args)
    shape = data_handler.get_dims('x', 'y', 'c')
    dim_l = data_handler.get_dims('labels')[0]
    net = Net(shape, dim_out=dim_l, **args)
    return dict(classifier=net), classify
