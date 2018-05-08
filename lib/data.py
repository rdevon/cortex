'''Data module

'''

import copy
import logging
import shutil
import signal
import os
from os import path

import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer, ETA
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision.datasets import utils
import torchvision.transforms as transforms

from . import config, exp
#from .cub import CUB


logger = logging.getLogger('cortex.data')

IMAGE_SCALE = [0, 1]


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


def make_transform(source, normalize=True, center_crop=None, image_size=None, random_crop=None, flip=None):
    default_normalization = {
        'MNIST': [(0.5,), (0.5,)],
        'Fashion-MNIST': [(0.5,), (0.5,)],
        'EMNIST': [(0.5,), (0.5,)],
        'PhotoTour': [(0.5,), (0.5,)],
        'Imagenet-12': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'LSUN': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'SVHN': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'CIFAR10': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'CIFAR100': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'CUB': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'CelebA': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'STL10': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    }

    global IMAGE_SCALE
    transform_ = []

    if random_crop:
        transform_.append(transforms.RandomSizedCrop(random_crop))
    elif center_crop:
        transform_.append(transforms.CenterCrop(image_crop))

    if image_size:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_.append(transforms.Resize(image_size))

    if flip:
        if isinstance(flip, bool):
            flip = 0.5
        torchvision.transforms.RandomHorizontalFlip()

    transform_.append(transforms.ToTensor())

    if normalize and isinstance(normalize, bool):
        if source in default_normalization.keys():
            normalize = default_normalization[source]
        else:
            raise ValueError('Default normalization for source {} not found. Please enter custom normalization.'
                             ''.format(source))

    transform_.append(transforms.Normalize(*normalize))
    if normalize[0] == (0.5, 0.5, 0.5):
        IMAGE_SCALE = [-1, 1]

    transform = transforms.Compose(transform_)
    return transform


def make_indexing(C):
    class IndexingDataset(C):
        def __getitem__(self, index):
            output = super().__getitem__(index)
            return output + (index,)

    return IndexingDataset


def copy_to_local_path(from_path):
    if from_path.endswith('/'):
        from_path = from_path[:-1]
    basename = path.basename(from_path)
    if not config.LOCAL_PATH:
        raise ValueError('`local_path` not set in `config.yaml`. Set this path if you want local copying.')
    to_path = path.join(config.LOCAL_PATH, basename)
    if ((not path.exists(to_path)) and path.exists(from_path)):
        logger.info('Copying {} to {}'.format(from_path, to_path))
        if path.isdir(from_path):
            shutil.copytree(from_path, to_path)
        else:
            shutil.copy(from_path, config.LOCAL_PATH)

    return to_path


class DataHandler(object):
    def __init__(self):
        self.dims = {}
        self.input_names = {}
        self.noise = {}
        self.loaders = {}
        self.batch = None
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

    def add_dataset(self, source, test_on_train, n_workers=4, duplicate=None, shuffle=True, copy_to_local=False,
                    DataLoader=None, Dataset=None, transform=None, **transform_args):

        DataLoader = DataLoader or torch.utils.data.DataLoader

        if hasattr(torchvision.datasets, source):
            # Dataset is in torchvision
            source_type = 'torchvision'
            if config.TV_PATH is None:
                raise ValueError('torchvision dataset must have corresponding torchvision folder specified in '
                                 '`config.yaml`')
            Dataset = getattr(torchvision.datasets, source)

            if copy_to_local:
                copy_to_local_path(path.join(config.TV_PATH, source))
                base_path = config.LOCAL_PATH
            else:
                base_path = config.TV_PATH

            train_path = path.join(base_path, source)
            test_path = train_path
        elif source in config.DATA_PATHS:
            # Dataset is specified in config.DATA_PATHS
            source_type = 'folder'
            data_path = config.DATA_PATHS[source]
            if isinstance(data_path, dict):
                train_path = data_path['train']
                test_path = data_path['test']
                if copy_to_local:
                    train_path = copy_to_local_path(train_path)
                    test_path = copy_to_local_path(test_path)
            elif isinstance(data_path, (tuple, list)):
                train_path, test_path = data_path
                if copy_to_local:
                    train_path = copy_to_local_path(train_path)
                    test_path = copy_to_local_path(test_path)
            else:
                train_path = data_path
                if copy_to_local:
                    train_path = copy_to_local_path(train_path)
                test_path = data_path

        elif path.isdir(source):
            # Dataset is a path to a folder
            source_type = 'folder'
            logger.info('Using train set as testing set. For more options, use `data_paths` in `config.yaml`')
            Dataset = Dataset or torchvision.datasets.ImageFolder
            train_path = source
            test_path = source
        else:
            raise ValueError('Dataset not from torchvision, or is not specified in `config.yaml` data_paths.')

        transform = make_transform(source, **transform_args)
        if source == 'CelebA':
            Dataset = Dataset or CelebA
            Dataset = make_indexing(Dataset)
            train_set = Dataset(root=train_path, transform=transform, download=True)
            test_set = Dataset(root=test_path, transform=transform)
            output_sources = ['images', 'targets']
        elif source == 'CUB':
            Dataset = Dataset or CUB
            Dataset = make_indexing(Dataset)
            train_set = Dataset(root=train_path, transform=transform, split_type='train')
            test_set = Dataset(root=test_path, transform=transform, split_type='test')
            output_sources = ['images', 'targets', 'attributes']
        elif source == 'SVHN':
            Dataset = make_indexing(Dataset)
            train_set = Dataset(train_path, split='train', transform=transform, download=True)
            test_set = Dataset(test_path, split='test', transform=transform, download=True)
            output_sources = ['images', 'targets']
        elif source == 'LSUN':
            Dataset = make_indexing(Dataset)
            train_set = Dataset(train_path, classes=['bedroom_train'], transform=transform)
            test_set = Dataset(test_path, classes=['bedroom_test'], transform=transform)
            output_sources = ['images', 'targets']
        elif source_type == 'folder':
            Dataset = Dataset or torchvision.datasets.ImageFolder
            Dataset = make_indexing(Dataset)
            train_set = Dataset(root=train_path, transform=transform)
            test_set = Dataset(root=test_path, transform=transform)
            output_sources = ['images', 'targets']
        elif Dataset:
            Dataset = make_indexing(Dataset)
            train_set = Dataset(root=train_path, train=True, download=True, transform=transform)
            test_set = Dataset(root=test_path, train=False, download=True, transform=transform)
            output_sources = ['images', 'targets']
        else:
            raise Exception

        output_sources += ['index']

        if test_on_train:
            test_set = copy.deepcopy(train_set)

        self.image_scale = IMAGE_SCALE # A bit of a hack. Needs to be fixed.

        N_train = len(train_set)
        N_test = len(test_set)

        train_loader = DataLoader(train_set, batch_size=self.batch_size['train'], shuffle=shuffle, num_workers=n_workers,
                                  worker_init_fn=lambda x: signal.signal(signal.SIGINT, signal.SIG_IGN))
        test_loader = DataLoader(test_set, batch_size=self.batch_size['test'], shuffle=shuffle, num_workers=n_workers,
                                 worker_init_fn=lambda x: signal.signal(signal.SIGINT, signal.SIG_IGN))

        if source_type == 'folder':
            for sample in train_loader:
                break
            dim_c, dim_x, dim_y = sample[0].size()[1:]
            dim_l = len(train_set.classes)
        elif source == 'SVHN':
            dim_c, dim_x, dim_y = train_set.data.shape[1:]
            dim_l = len(np.unique(train_set.labels))
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

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l, n_train=N_train, n_test=N_test)
        if source == 'CUB':
            dim_a = train_set.attrs.shape[1]
            dims['a'] = dim_a

        logger.debug('Data has the following dimensions: {}'.format(dims))
        if not duplicate:
            self.dims[source] = dims
            self.input_names[source] = output_sources
            self.loaders.update(**{source: dict(train=train_loader, test=test_loader)})
            self.sources.append(source)
        else:
            for i in range(duplicate):
                source_ = source + '_{}'.format(i)
                self.dims[source_] = dims
                self.input_names[source_] = output_sources
                self.loaders.update(**{source_: dict(train=train_loader, test=test_loader)})
                self.sources.append(source_)

    def add_noise(self, key, dist=None, size=None, **kwargs):
        if size is None:
            raise ValueError

        dim = size

        if not isinstance(size, tuple):
            size = (size,)

        train_size = (self.batch_size['train'],) + size
        test_size = (self.batch_size['test'],) + size

        def expand_train(*args):
            return (torch.zeros(train_size) + a for a in args)

        def expand_test(*args):
            return (torch.zeros(test_size) + a for a in args)

        if dist == 'bernoulli':
            Dist = torch.distributions.bernoulli.Bernoulli
            raise NotImplementedError(dist)
        elif dist == 'beta':
            Dist = torch.distributions.beta.Beta
            raise NotImplementedError(dist)
        elif dist == 'binomial':
            Dist = torch.distributions.binomial.Binomial
            raise NotImplementedError(dist)
        elif dist == 'categorical':
            Dist = torch.distributions.categorical.Categorical
            raise NotImplementedError(dist)
        elif dist == 'cauchy':
            Dist = torch.distributions.cauchy.Cauchy
            raise NotImplementedError(dist)
        elif dist == 'chi2':
            Dist = torch.distributions.chi2.Chi2
            raise NotImplementedError(dist)
        elif dist == 'dirichlet':
            Dist = torch.distributions.dirichlet.Dirichlet
            conc = kwargs.pop('concentration', 1.)
            conc_tr, = expand_train(conc)
            conc_te, = expand_test(conc)
            var = Dist(conc_tr, **kwargs)
            var_t = Dist(conc_te, **kwargs)
        elif dist == 'exponential':
            Dist = torch.distributions.exponential.Exponential
            raise NotImplementedError(dist)
        elif dist == 'fishersnedecor':
            Dist = torch.distributions.fishersnedecor.FisherSnedecor
            raise NotImplementedError(dist)
        elif dist == 'gamma':
            Dist = torch.distributions.gamma.Gamma
            raise NotImplementedError(dist)
        elif dist == 'geometric':
            Dist = torch.distributions.geometric.Geometric
            raise NotImplementedError(dist)
        elif dist == 'gumbel':
            Dist = torch.distributions.gumbel.Gumbel
            loc = kwargs.pop('loc', 0)
            scale = kwargs.pop('scale', 1)
            raise NotImplementedError(dist)
        elif dist == 'laplace':
            Dist = torch.distributions.laplace.Laplace
            loc = kwargs.pop('loc', 0)
            scale = kwargs.pop('scale', 1)
            raise NotImplementedError(dist)
        elif dist == 'log_normal':
            Dist = torch.distributions.log_normal.LogNormal
            loc = kwargs.pop('loc', 0)
            scale = kwargs.pop('scale', 1)
            raise NotImplementedError(dist)
        elif dist == 'multinomial':
            Dist = torch.distributions.multinomial.Multinomial
            raise NotImplementedError(dist)
        elif dist == 'multivariate_normal':
            Dist = torch.distributions.multivariate_normal.MultivariateNormal
            raise NotImplementedError(dist)
        elif dist == 'normal':
            Dist = torch.distributions.normal.Normal
            loc = kwargs.pop('loc', 0.)
            scale = kwargs.pop('scale', 1.)
            loc_tr, scale_tr = expand_train(loc, scale)
            loc_te, scale_te = expand_test(loc, scale)

            var = Dist(loc_tr, scale_tr, **kwargs)
            var_t = Dist(loc_te, scale_te, **kwargs)

        elif dist == 'one_hot_categorical':
            Dist = torch.distributions.one_hot_categorical.OneHotCategorical
            raise NotImplementedError(dist)
        elif dist == 'pareto':
            Dist = torch.distributions.pareto.Pareto
            raise NotImplementedError(dist)
        elif dist == 'poisson':
            Dist = torch.distributions.poisson.Poisson
            raise NotImplementedError(dist)
        elif dist == 'relaxed_bernoulli':
            Dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli
            raise NotImplementedError(dist)
        elif dist == 'relaxed_categorical':
            Dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical
            raise NotImplementedError(dist)
        elif dist == 'studentT':
            Dist = torch.distributions.studentT.StudentT
            raise NotImplementedError(dist)
        elif dist == 'uniform':
            Dist = torch.distributions.uniform.Uniform
            low = kwargs.pop('low', 0.)
            high = kwargs.pop('high', 1.)

            low_tr, high_tr = expand_train(low, high)
            low_te, high_te = expand_test(low, high)

            var = Dist(low_tr, high_tr, **kwargs)
            var_t = Dist(low_te, high_te, **kwargs)

        else:
            raise NotImplementedError('`{}` distribution not found'.format(dist))

        d_args = dict(
            beta=['concentration1', 'concentration0'],
            cachy=['loc', 'scale'],
            chi2=['df'],
            dirichlet=['concentration'],
            exponential=['rate'],
            fishersnedecor=['df1', 'df2'],
            gamma=['concentration', 'rate'],
            gumbel=['loc', 'scale'],
            laplace=['loc', 'scale'],
            log_normal=['loc', 'scale'],
            multivariate_normal=['loc'],
            normal=['loc', 'scale'],
            pareto=['scale', 'alpha'],
            poisson=['rate'],
            relaxed_bernoulli=['temperature'],
            relaxed_categorical=['temperature'],
            studentT=['df'],
            uniform=['high', 'low']
        )

        self.noise[key] = (var, var_t)
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

        for k, (n_var, n_var_t) in self.noise.items():
            if self.mode == 'test':
                n_var = n_var_t

            n_var = n_var.sample()

            if exp.USE_CUDA:
                n_var = n_var.to('cuda')

            if n_var.size()[0] != batch_size:
                n_var = n_var[0:batch_size]
            output[k] = n_var

        self.batch = output
        self.u += 1
        self.update_pbar()

        return self.batch

    def next(self):
        return self.__next__()

    def __getitem__(self, item):
        if self.batch is None:
            raise RuntimeError('Batch not set')

        if not item in self.batch.keys():
            raise KeyError('Data with label `{}` not found. Available: {}'.format(item, tuple(self.batch.keys())))
        batch = self.batch[item]

        return batch

    def get_batch(self, *item):
        if self.batch is None:
            raise RuntimeError('Batch not set')

        batch = []
        for i in item:
            if '.' in i:
                j, i_ = i.split('.')
                j = int(j)
                batch.append(self.batch[list(self.batch.keys())[j-1]][i_])
            elif not i in self.batch.keys():
                raise KeyError('Data with label `{}` not found. Available: {}'.format(i, tuple(self.batch.keys())))
            else:
                batch.append(self.batch[i])
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def get_dims(self, *q):
        if q[0] in self.dims.keys():
            dims = self.dims
        else:
            key = [k for k in self.dims.keys() if k not in self.noise.keys()][0]
            dims = self.dims[key]

        try:
            d = [dims[q_] for q_ in q]
        except KeyError:
            raise KeyError('Cannot resolve dimensions {}, provided {}'.format(q, dims))
        if len(d) == 1:
            return d[0]
        else:
            return d

    def make_iterator(self, source):
        loader = self.loaders[source][self.mode]

        def iterator():
            for inputs in loader:
                if exp.USE_CUDA:
                    inputs = [inp.to('cuda') for inp in inputs]
                inputs_ = []

                for i, inp in enumerate(inputs):
                    inputs_.append(inp)

                yield inputs_

        return iterator()

    def update_pbar(self):
        if self.pbar:
            self.pbar.update(self.u)

    def reset(self, test=False, make_pbar=True, string=''):
        self.mode = 'test' if test else 'train'
        self.u = 0

        if make_pbar:
            widgets = [string, Timer(), ' | ', Percentage(), ' | ', ETA(), Bar()]
            if len([len(loader[self.mode]) for loader in self.loaders.values()]) == 0:
                maxval = 1000
            else:
                maxval = min(len(loader[self.mode]) for loader in self.loaders.values())
            self.pbar = ProgressBar(widgets=widgets, maxval=maxval).start()
        else:
            self.pbar = None

        self.iterators = dict((source, self.make_iterator(source)) for source in self.sources)

DATA_HANDLER = DataHandler()


def setup(source=None, batch_size=64, noise_variables=None, n_workers=4, skip_last_batch=False,
          test_on_train=False, setup_fn=None, **kwargs):
    global DATA_HANDLER, NOISE

    if source and not isinstance(source, (list, tuple)):
        source = [source]

    DATA_HANDLER.set_batch_size(batch_size, skip_last_batch=skip_last_batch)

    if source:
        for source_ in source:
            source_args = kwargs.get(source_, kwargs)
            DATA_HANDLER.add_dataset(source_, test_on_train, n_workers=n_workers, **source_args)

    if noise_variables:
        for k, v in noise_variables.items():
            DATA_HANDLER.add_noise(k, **v)