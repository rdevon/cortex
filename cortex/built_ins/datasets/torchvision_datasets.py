'''Entrypoint for torchvision datasets.

'''

import os

import numpy as np
import torchvision
from torchvision.transforms import transforms

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

    def _handle_LSUN(self, Dataset, data_path, transform=None, **kwargs):
        train_set = Dataset(
            data_path,
            classes=['bedroom_train'],
            transform=transform)
        test_set = Dataset(
            data_path,
            classes=['bedroom_test'],
            transform=transform)
        return train_set, test_set

    def _handle_SVHN(self, Dataset, data_path, transform=None, **kwargs):
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

    def _handle_STL(self, Dataset, data_path, transform=None,
                    labeled_only=False):
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            normalize,
        ])
        if labeled_only:
            split = 'train'
        else:
            split = 'train+unlabeled'
        train_set = Dataset(
            data_path,
            split=split,
            transform=train_transform,
            download=True
        )
        test_set = Dataset(
            data_path,
            split='test',
            transform=test_transform,
            download=True)
        return train_set, test_set

    def _handle(self, Dataset, data_path, transform=None, **kwargs):
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

    def handle(self, source, copy_to_local=False, normalize=True,
               train_samples=None, test_samples=None,
               labeled_only=False, **transform_args):

        Dataset = getattr(torchvision.datasets, source)
        Dataset = self.make_indexing(Dataset)

        torchvision_path = self.get_path('torchvision')
        if not os.path.isdir(torchvision_path):
            os.mkdir(torchvision_path)

        data_path = os.path.join(torchvision_path, source)

        if copy_to_local:
            data_path = self.copy_to_local_path(data_path)

        if normalize and isinstance(normalize, bool):
            if source in ['MNIST', 'dSprites', 'Fashion-MNIST', 'EMNIST',
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
        elif source == 'STL10':
            handler = self._handle_STL
        else:
            handler = self._handle

        train_set, test_set = handler(Dataset, data_path, transform=transform,
                                      labeled_only=labeled_only)
        if train_samples is not None:
            train_set.train_data = train_set.train_data[:train_samples]
            train_set.train_labels = train_set.train_labels[:train_samples]
        if test_samples is not None:
            test_set.test_data = test_set.test_data[:test_samples]
            test_set.test_labels = test_set.test_labels[:test_samples]

        if source in ('SVHN', 'STL10'):
            dim_c, dim_x, dim_y = train_set[0][0].size()
            uniques = np.unique(train_set.labels).tolist()
            try:
                uniques.remove(-1)
            except ValueError:
                pass
            dim_l = len(uniques)
        else:
            if len(train_set.train_data.shape) == 4:
                dim_c, dim_x, dim_y = train_set[0][0].size()
            else:
                dim_x, dim_y = train_set[0][0].size()
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
