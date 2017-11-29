'''Data module

'''

import logging
from os import path

import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import config
import exp

logger = logging.getLogger('cortex.data')


LOADERS = {}
DIMS = {}


def make_iterator(test=False, make_pbar=True, string=''):
    volatile = test

    if test:
        loader = LOADERS['test']
    else:
        loader = LOADERS['train']

    if make_pbar:
        widgets = [string, Timer(), Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=len(loader)).start()
    else:
        pbar = None

    def iterator(u=0):
        for u, inputs in enumerate(loader):
            if exp.USE_CUDA:
                inputs = [inp.cuda() for inp in inputs]
            inputs_ = []
            for i, inp in enumerate(inputs):
                if i == 0:
                    inputs_.append(Variable(inp, volatile=volatile))
                else:
                    inputs_.append(Variable(inp))
            inputs = tuple(inputs_)
            if pbar:
                pbar.update(u)
            yield inputs

    return iterator()


def setup(source=None, batch_size=None, test_batch_size=None, n_workers=4, meta=None,
          normalize=True, scale=None):
    global LOADERS, DIMS

    if hasattr(torchvision.datasets, source):
        dataset = getattr(torchvision.datasets, source)

    if normalize:
        if source == 'MNIST':
            norm = transforms.Normalize((0.1307,), (0.3081,))
        else:
            #norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(), norm])

    if not source:
        raise ValueError('Source not provided.')
    else:
        source = path.join(config.DATA_PATH, source)
    if not batch_size:
        raise ValueError('Batch size not provided.')
    test_batch_size = test_batch_size or batch_size

    logger.info('Loading data from `{}`'.format(source))
    train_set = dataset(root=source, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    test_set = dataset(root=source, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              num_workers=n_workers)

    n_train, dim_x, dim_y, dim_c = tuple(train_set.train_data.shape)
    dim_l = len(np.unique(train_set.train_labels))
    n_test = test_set.test_data.shape[0]
    DIMS.update(n_train=n_train, n_test=n_test, dim_x=dim_x, dim_y=dim_y, dim_c=dim_c, dim_l=dim_l)
    logger.debug('Data has the following dimensions: {}'.format(DIMS))

    LOADERS.update(train=train_loader, test=test_loader)