'''Simple classifier model

'''

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger('cortex.models' + __name__)

GLOBALS = {'DIM_X': None, 'DIM_Y': None, 'DIM_C': None, 'DIM_Z': None}


DEFAULTS = dict(
    data=dict(batch_size=128),
    optimizer=dict(
        optimizer='SGD',
        learning_rate=1e-2,
    ),
    model=dict(loss=nn.CrossEntropyLoss()),
    train=dict(
        epochs=200,
        summary_updates=100,
        archive_every=10
    )
)


def results(net, inputs, criterion):
    images, targets = inputs
    outputs = net(images)
    loss = criterion(outputs, targets)
    _, predicted = torch.max(outputs.data, 1)
    correct = 100. * predicted.eq(targets.data).cpu().sum() / targets.size(0)
    return loss, dict(loss=loss.data[0], accuracy=correct), 'accuracy'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def build_model(loss=None):
    net = Net()
    return dict(classifier=net), dict(classifier=loss), dict(classifier=results)


