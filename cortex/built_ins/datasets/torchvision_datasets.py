'''Entrypoint for torchvision datasets.

'''

import os

import numpy as np
import torchvision

from cortex.plugins import DatasetPlugin, register_plugin
from .utils import build_transforms


class TorchvisionDatasetPlugin(DatasetPlugin):
    sources = [
        'CIFAR10',
        'CIFAR100',
        'CocoCaptions',
        'CocoDetection',
        'FakeData',
        'FashionMNIST',
        'ImageFolder',
        'LSUN',
        'LSUNClass',
        'MNIST',
        'PhotoTour',
        'SEMEION',
        'STL10',
        'SVHN']

    def _handle_LSUN(self, Dataset, data_path, transform=None):
        train_set = Dataset(
            data_path,
            classes=['bedroom_train'],
            transform=transform)
        test_set = Dataset(
            data_path,
            classes=['bedroom_test'],
            transform=transform)
        return train_set, test_set

    def _handle_SVHN(self, Dataset, data_path, transform=None):
        train_set = Dataset(
            data_path,
            split='train',
            transform=transform,
            download=True)
        test_set = Dataset(
            data_path,
            split='test',
            transform=transform,
            download=True)
        return train_set, test_set

    def _handle(self, Dataset, data_path, transform=None):
        train_set = Dataset(
            data_path,
            train=True,
            transform=transform,
            download=True)
        test_set = Dataset(
            data_path,
            train=False,
            transform=transform,
            download=True)
        return train_set, test_set

    def handle(
            self,
            source,
            copy_to_local=False,
            normalize=True,
            **transform_args):

        Dataset = getattr(torchvision.datasets, source)
        Dataset = self.make_indexing(Dataset)
        torchvision_path = self.get_path('torchvision')
        if not os.path.isdir(torchvision_path):
            os.mkdir(torchvision_path)

        data_path = os.path.join(torchvision_path, source)

        if copy_to_local:
            data_path = self.copy_to_local_path(data_path)

        if normalize and isinstance(normalize, bool):
            if source in [
                'MNIST',
                'dSprites',
                'Fashion-MNIST',
                'EMNIST',
                    'PhotoTour']:
                normalize = [(0.5,), (0.5,)]
                scale = (0, 1)
            else:
                normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
                scale = (-1, 1)

        else:
            scale = None

        transform = build_transforms(normalize=normalize, **transform_args)

        if source == 'LSUN':
            handler = self._handle_LSUN
        elif source == 'SVHN':
            handler = self._handle_SVHN
        else:
            handler = self._handle

        train_set, test_set = handler(Dataset, data_path, transform=transform)

        if source == 'SVHN':
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

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l)
        input_names = ['images', 'targets', 'index']

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        if scale is not None:
            self.set_scale(scale)


register_plugin(TorchvisionDatasetPlugin)
