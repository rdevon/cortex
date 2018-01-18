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
INPUT_NAMES = []
NOISE = {}


def make_iterator(test=False, make_pbar=True, string=''):

    if test:
        loader = LOADERS['test']
    else:
        loader = LOADERS['train']

    if make_pbar:
        widgets = [string, Timer(), Bar()]
        pbar = ProgressBar(widgets=widgets, maxval=len(loader)).start()
    else:
        pbar = None

    def iterator():
        for u, inputs in enumerate(loader):

            if exp.USE_CUDA:
                inputs = [inp.cuda() for inp in inputs]
            inputs_ = []

            for i, inp in enumerate(inputs):
                if i == 0:
                    inputs_.append(Variable(inp, volatile=test))
                else:
                    inputs_.append(Variable(inp))

            if len(NOISE) > 0:
                noise = [NOISE[k] for k in INPUT_NAMES if k in NOISE.keys()]
                for (n_var, dist) in noise:
                    if dist == 'normal':
                        n_var = n_var.normal_(0, 1)
                    elif dist == 'uniform':
                        n_var = n_var.uniform_(0, 1)
                    else:
                        raise NotImplementedError(dist)
                    if n_var.size()[0] != inputs[0].size()[0]:
                        n_var = n_var[0:inputs[0].size()[0]]
                    inputs_.append(Variable(n_var, volatile=test))

            inputs = dict(zip(INPUT_NAMES, inputs_))
            if pbar:
                pbar.update(u)
            yield inputs

    return iterator()


def setup(source=None, batch_size=None, test_batch_size=1000, n_workers=4, meta=None,
          normalize=True, image_size=None, image_crop=None, noise_variables=None,
          test_on_train=False):
    global LOADERS, DIMS, INPUT_NAMES, NOISE

    if not source:
        raise ValueError('Source not provided.')

    if path.isdir(source):
        logger.info('Using train set as testing set. For more options, use `data_paths` in `config.yaml`')
        isfolder = True
        dataset = torchvision.datasets.ImageFolder
        train_path = source
        test_path = source
    elif hasattr(torchvision.datasets, source):
        isfolder = (source == 'Imagenet-12')
        dataset = getattr(torchvision.datasets, source)
        if config.TV_PATH is None:
            raise ValueError('torchvision dataset must have corresponding torchvision folder specified in `config.yaml`')
        train_path = path.join(config.TV_PATH, source)
        test_path = train_path
    else:
        isfolder = True
        dataset = torchvision.datasets.ImageFolder
        if source not in config.DATA_PATHS.keys():
            raise ValueError('Custom dataset not specified in `config.yaml` data_paths.')
        if isinstance(config.DATA_PATHS[source], dict):
            train_path = path.join(config.DATA_PATHS[source]['train'])
            test_path = path.join(config.DATA_PATHS[source]['test'])
        else:
            test_path = path.join(config.DATA_PATHS[source])

    transform_ = []
    if isfolder:
        transform_.append(transforms.RandomSizedCrop(224))
        image_size = (64, 64)

    if image_size:
        transform_.append(transforms.Resize(image_size))

    if image_crop:
        transform_.append(transforms.CenterCrop(image_crop))

    transform_.append(transforms.ToTensor())

    if normalize:
        if source == 'MNIST':
            transform_.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            #norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transform_.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    transform = transforms.Compose(transform_)

    if not batch_size:
        raise ValueError('Batch size not provided.')
    test_batch_size = test_batch_size or batch_size

    logger.info('Loading data from `{}`'.format(source))

    if source == 'LSUN':
        train_set = dataset(train_path, classes=['bedroom_train'], transform=transform)
        if test_on_train:
            test_set = train_set
        else:
            test_set = dataset(test_path, classes=['bedroom_test'], transform=transform)
    elif isfolder:
        train_set = dataset(root=train_path, transform=transform)
        if test_on_train:
            test_set = train_set
        else:
            test_set = dataset(root=test_path, transform=transform)
    else:
        train_set = dataset(root=train_path, train=True, download=True, transform=transform)
        if test_on_train:
            test_set = train_set
        else:
            test_set = dataset(root=test_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=n_workers)

    if isfolder:
        for sample in train_loader:
            break
        dim_c, dim_x, dim_y = sample[0].size()[1:]
        dim_l = len(train_set.classes)
    else:
        if len(train_set.train_data.shape) == 4:
            dim_x, dim_y, dim_c = tuple(train_set.train_data.shape)[1:]
        else:
            dim_x, dim_y = tuple(train_set.train_data.shape)[1:]
            dim_c = 1
        dim_l = len(np.unique(train_set.train_labels))

    DIMS.update(dim_x=dim_x, dim_y=dim_y, dim_c=dim_c, dim_l=dim_l)
    logger.debug('Data has the following dimensions: {}'.format(DIMS))
    INPUT_NAMES = ['images', 'targets']

    if noise_variables:
        for k, (dist, dim) in noise_variables.items():
            var = torch.FloatTensor(batch_size, dim)
            if exp.USE_CUDA:
                var = var.cuda()
            NOISE[k] = (var, dist)
            INPUT_NAMES.append(k)

    LOADERS.update(train=train_loader, test=test_loader)