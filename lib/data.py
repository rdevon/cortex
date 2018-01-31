'''Data module

'''

import logging
import os
from os import path

import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision.datasets import utils
import torchvision.transforms as transforms

from . import config, exp


logger = logging.getLogger('cortex.data')

_default_normalization = {
    'MNIST': [(0.5,), (0.5,)],
    'Fashion-MNIST': [(0.5,), (0.5,)],
    'EMNIST': [(0.5,), (0.5,)],
    'PhotoTour': [(0.5,), (0.5,)],
    'Imagenet-12': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'LSUN': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'CIFAR10': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'CIFAR100': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    'STL10': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
}


class CelebA(torchvision.datasets.ImageFolder):
    url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1"
    filename = "img_align_celeba.zip"

    def __init__(self, root, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        super(CelebA, self).__init__(root, transform, target_transform)

    def download(self):
        import errno
        import zipfile
        from six.moves import urllib

        root = self.root
        url = self.url

        root = os.path.expanduser(root)
        fpath = os.path.join(root, self.filename)
        image_dir = os.path.join(root, 'images')

        try:
            os.makedirs(image_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                logger.info('Dataset exists, not downloading.')
                return
            else:
                raise

        # downloads file
        if os.path.isfile(fpath):
            logger.info('Using downloaded file: {}'.format(fpath))
        else:
            try:
                logger.info('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    logger.info('Failed download. Trying https -> http instead.'
                                ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(url, fpath)
                else:
                    raise

        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(image_dir)
        zip_ref.close()


def make_transform(source, normalize=True, image_crop=None, image_size=None, isfolder=False):
    transform_ = []

    if isfolder:
        if source != 'CelebA':
            transform_.append(transforms.RandomSizedCrop(224))
        image_size = (64, 64)
        normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

    if image_size:
        transform_.append(transforms.Resize(image_size))

    if image_crop:
        transform_.append(transforms.CenterCrop(image_crop))

    transform_.append(transforms.ToTensor())

    if normalize and isinstance(normalize, bool):
        if source in _default_normalization.keys():
            transform_.append(transforms.Normalize(*_default_normalization[source]))
        else:
            raise ValueError('Default normalization for source {} not found. Please enter custom normalization.'
                             ''.format(source))
    else:
        transform_.append(transforms.Normalize(*normalize))

    transform = transforms.Compose(transform_)
    return transform


class DataHandler(object):
    def __init__(self):
        self.dims = {}
        self.input_names = {}
        self.noise = {}
        self.loaders = {}
        self.batch = {}
        self.noise = {}
        self.iterator = {}
        self.sources = []
        self.pbar = None
        self.u = 0

    def set_batch_size(self, batch_size, skip_last_batch=False):
        if not isinstance(batch_size, dict):
            self.batch_size = dict(train=batch_size)
        else:
            self.batch_size = batch_size
        if 'test' not in self.batch_size.keys():
            self.batch_size['test'] = self.batch_size['train']
        self.skip_last_batch = skip_last_batch

    def add_dataset(self, source, test_on_train, n_workers=4, **source_args):
        if path.isdir(source):
            logger.info('Using train set as testing set. For more options, use `data_paths` in `config.yaml`')
            source_type = 'folder'
            dataset = torchvision.datasets.ImageFolder
            train_path = source
            test_path = source
        elif hasattr(torchvision.datasets, source):
            source_type = 'torchvision'

            if config.TV_PATH is None:
                raise ValueError(
                    'torchvision dataset must have corresponding torchvision folder specified in `config.yaml`')
            train_path = path.join(config.TV_PATH, source)
            test_path = train_path
        else:
            if source not in config.DATA_PATHS.keys():
                raise ValueError('Custom dataset not specified in `config.yaml` data_paths.')
            if isinstance(config.DATA_PATHS[source], dict):
                train_path = path.join(config.DATA_PATHS[source]['train'])
                test_path = path.join(config.DATA_PATHS[source]['test'])
            else:
                train_path = path.join(config.DATA_PATHS[source])
                test_path = path.join(config.DATA_PATHS[source])
            source_type = 'folder'

        if source_type == 'torchvision':
            dataset = getattr(torchvision.datasets, source)
        elif source_type == 'folder':
            if source == 'CelebA':
                dataset = CelebA
            else:
                dataset = torchvision.datasets.ImageFolder

        transform = make_transform(source, isfolder=(source_type=='folder'), **source_args)

        if source == 'LSUN':
            train_set = dataset(train_path, classes=['bedroom_train'], transform=transform)
            if test_on_train:
                test_set = train_set
            else:
                test_set = dataset(test_path, classes=['bedroom_test'], transform=transform)
        elif source_type == 'folder':
            if source == 'CelebA':
                train_set = dataset(root=train_path, transform=transform, download=True)
            else:
                train_set = dataset(root=train_path, transform=transform)
            if test_on_train:
                test_set = train_set
            else:
                test_set = dataset(root=test_path, transform=transform)
        elif source_type == 'hdf5':
            train_set = dataset(train_path, train=True, transform=transform)
            test_set = dataset(train_path, train=True, transform=transform)
        else:
            train_set = dataset(root=train_path, train=True, download=True, transform=transform)
            if test_on_train:
                test_set = train_set
            else:
                test_set = dataset(root=test_path, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size['train'], shuffle=True,
                                                   num_workers=n_workers)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size['test'], shuffle=True,
                                                  num_workers=n_workers)

        if source_type == 'folder':
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

            labels = train_set.train_labels
            if not isinstance(labels, list):
                labels = labels.numpy()
            dim_l = len(np.unique(labels))

        self.dims[source] = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l)
        logger.debug('Data has the following dimensions: {}'.format(self.dims[source]))
        self.input_names[source] = ['images', 'targets']
        self.loaders.update(**{source: dict(train=train_loader, test=test_loader)})
        self.sources.append(source)

    def add_noise(self, key, dist, dim):
        var = torch.FloatTensor(self.batch_size['train'], dim)
        var_t = torch.FloatTensor(self.batch_size['test'], dim)
        if exp.USE_CUDA:
            var = var.cuda()
            var_t = var_t.cuda()
        self.noise[key] = (var, var_t, dist)
        self.dims[key] = dim

    def get_label_names(self, source=None):
        source = source or self.sources[0]
        if source == 'CIFAR10':
            names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            names = ['{}'.format(i) for i in range(self.dims[source]['labels'])]
        return names

    def __iter__(self):
        return self

    def __next__(self):
        output = {}

        batch_size = self.batch_size[self.mode]
        for source in self.sources:
            data = next(self.iterators[source])
            if data[0].size()[0] < batch_size:
                if self.skip_last_batch:
                    raise StopIteration
                batch_size = data[0].size()[0]
            data = dict((k, v) for k, v in zip(self.input_names[source], data))
            if len(self.sources) > 1:
                output[source] = data
            else:
                output.update(**data)

        for k, (n_var, n_var_t, dist) in self.noise.items():
            if self.mode == 'test':
                n_var = n_var_t

            if dist == 'normal':
                n_var = n_var.normal_(0, 1)
            elif dist == 'uniform':
                n_var = n_var.uniform_(0, 1)
            else:
                raise NotImplementedError(dist)

            if n_var.size()[0] != batch_size:
                n_var = n_var[0:batch_size]

            output[k] = Variable(n_var, volatile=(self.mode=='test'))

        self.batch = output
        self.u += 1
        self.update_pbar()

        return self.batch

    def next(self):
        return self.__next__()

    def __getitem__(self, item):
        if self.batch is None:
            raise KeyError('Batch not set')
        return self.batch[item]

    def get_dims(self, *q):
        if q[0] in self.dims.keys():
            if len(q[1:]) != 0:
                dims = self.dims[q[0]]
                q = q[1:]
            else:
                dims = self.dims
        else:
            key = [k for k in self.dims.keys() if k not in self.noise.keys()][0]
            dims = self.dims[key]

        try:
            d = [dims[q_] for q_ in q]
        except KeyError:
            raise KeyError('Cannot resolve dimensions {}, provided {}'.format(q, dims))
        return d

    def make_iterator(self, source):
        loader = self.loaders[source][self.mode]

        def iterator():
            for inputs in loader:
                if exp.USE_CUDA:
                    inputs = [inp.cuda() for inp in inputs]
                inputs_ = []

                for i, inp in enumerate(inputs):
                    if i == 0:
                        inputs_.append(Variable(inp, volatile=(self.mode=='test')))
                    else:
                        inputs_.append(Variable(inp))

                yield inputs_

        return iterator()

    def update_pbar(self):
        if self.pbar:
            self.pbar.update(self.u)

    def reset(self, test=False, make_pbar=True, string=''):
        self.mode = 'test' if test else 'train'
        self.u = 0

        if make_pbar:
            widgets = [string, Timer(), Bar()]
            maxval = min(len(loader[self.mode]) for loader in self.loaders.values())
            self.pbar = ProgressBar(widgets=widgets, maxval=maxval).start()
        else:
            self.pbar = None

        self.iterators = dict((source, self.make_iterator(source)) for source in self.sources)

DATA_HANDLER = DataHandler()


def setup(source=None, batch_size=64, noise_variables=None, n_workers=4, skip_last_batch=False,
          test_on_train=False, setup_fn=None, **kwargs):
    global DATA_HANDLER, NOISE

    if not source:
        raise ValueError('Source not provided.')
    if not isinstance(source, (list, tuple)):
        source = [source]

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)

    for source_ in source:
        source_args = kwargs.get(source_, kwargs)
        DATA_HANDLER.add_dataset(source_, test_on_train, n_workers=n_workers, **source_args)

    if noise_variables:
        for k, (dist, dim) in noise_variables.items():
            DATA_HANDLER.add_noise(k, dist, dim)