'''Data module

'''

import logging
from os import path

from progressbar import Bar, ProgressBar, Percentage, Timer
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import config
import exp

logger = logging.getLogger('cortex.data')


LOADERS = {}


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


def setup(source=None, batch_size=None, test_batch_size=None, n_workers=4, meta=None):
    global LOADERS
    if hasattr(torchvision.datasets, source):
        dataset = getattr(torchvision.datasets, source)

    if not source:
        raise ValueError('Source not provided.')
    else:
        source = path.join(config.DATA_PATH, source)
    if not batch_size:
        raise ValueError('Batch size not provided.')
    test_batch_size = test_batch_size or batch_size

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    logger.info('Loading data from `{}`'.format(source))
    train_set = dataset(root=source, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

    test_set = dataset(root=source, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              num_workers=n_workers)

    LOADERS.update(train=train_loader, test=test_loader)