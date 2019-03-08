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
        'CIFAR10', 'CIFAR100', 'CocoCaptions', 'CocoDetection', 'FakeData',
        'FashionMNIST', 'ImageFolder', 'LSUN', 'LSUNClass', 'MNIST',
        'PhotoTour', 'SEMEION', 'STL10', 'SVHN'
    ]

    def _handle_LSUN(self, Dataset, data_path, transform=None, test_transform=None, **kwargs):
        train_set = Dataset(
            data_path, classes=['bedroom_train'], transform=transform)
        test_set = Dataset(
            data_path, classes=['bedroom_test'], transform=transform)
        return train_set, test_set

    def _handle_SVHN(self, Dataset, data_path, transform=None, test_transform=None, **kwargs):
        train_set = Dataset(
            data_path, split='train', transform=transform, download=True)
        test_set = Dataset(
            data_path, split='test', transform=test_transform, download=True)
        return train_set, test_set

    def _handle_STL(self, Dataset, data_path, transform=None, test_transform=None,
                    labeled_only=False):

        if labeled_only:
            split = 'train'
        else:
            split = 'train+unlabeled'
        train_set = Dataset(
            data_path, split=split, transform=transform, download=True)
        test_set = Dataset(
            data_path, split='test', transform=test_transform, download=True)
        return train_set, test_set

    def _handle(self, Dataset, data_path, transform=None, **kwargs):
        train_set = Dataset(
            data_path, train=True, transform=transform, download=True)
        test_set = Dataset(
            data_path, train=False, transform=transform, download=True)
        return train_set, test_set

    def handle(self, source, normalize=True,
               train_samples: int = None, test_samples: int = None,
               labeled_only=False, transform=None,
               center_crop: int = None, image_size: int = None,
               random_crop: int = None, flip=False, random_resize_crop: int = None,
               random_sized_crop: int = None,
               center_crop_test: int = None, image_size_test: int = None,
               random_crop_test: int = None, flip_test=False, random_resize_crop_test: int = None,
               random_sized_crop_test: int = None
               ):
        '''

        Args:
            normalize: Normalization of the image.
            train_samples: Number of training samples.
            test_samples: Number of test samples.
            labeled_only: Only use labeled data.
            transform: Transformation class object.
            center_crop: Center cropping of the image.
            image_size: Final size of the image.
            random_crop: Random cropping of the image.
            flip: Random flipping.
            random_resize_crop: Random resizing and cropping of the image.
            random_sized_crop: Random sizing and cropping of the image.

        '''

        Dataset = getattr(torchvision.datasets, source)
        Dataset = self.make_indexing(Dataset)

        torchvision_path = self.get_path('torchvision')
        if not os.path.isdir(torchvision_path):
            os.mkdir(torchvision_path)

        data_path = os.path.join(torchvision_path, source)

        if self.copy_to_local:
            data_path = self.copy_to_local_path(data_path)

        if transform is None:
            if normalize and isinstance(normalize, bool):
                if source in [
                    'MNIST', 'dSprites', 'Fashion-MNIST', 'EMNIST', 'PhotoTour'
                ]:
                    normalize = [(0.5,), (0.5,)]
                    scale = (0, 1)
                else:
                    normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
                    scale = (-1, 1)

            else:
                scale = None

            train_transform = build_transforms(
                normalize=normalize, center_crop=center_crop, image_size=image_size,
                random_crop=random_crop, flip=flip, random_resize_crop=random_resize_crop,
                random_sized_crop=random_sized_crop)
            test_transform = build_transforms(
                normalize=normalize, center_crop=center_crop_test, image_size=image_size_test,
                random_crop=random_crop_test, flip=flip_test, random_resize_crop=random_resize_crop_test,
                random_sized_crop=random_sized_crop_test)

        if source == 'LSUN':
            handler = self._handle_LSUN
        elif source == 'SVHN':
            handler = self._handle_SVHN
        elif source == 'STL10':
            handler = self._handle_STL
        else:
            handler = self._handle

        train_set, test_set = handler(Dataset, data_path, transform=train_transform, test_transform=test_transform,
                                      labeled_only=labeled_only)
        if train_samples is not None:
            train_set.train_data = train_set.train_data[:train_samples]
            train_set.train_labels = train_set.train_labels[:train_samples]
        if test_samples is not None:
            test_set.test_data = test_set.test_data[:test_samples]
            test_set.test_labels = test_set.test_labels[:test_samples]

        dim_images = train_set[0][0].size()

        if source in ('SVHN', 'STL10'):
            labels = train_set.labels
            uniques = np.unique(labels).tolist()
            try:
                uniques.remove(-1)
            except ValueError:
                pass
        else:
            labels = train_set.train_labels
            if not isinstance(labels, list):
                labels = labels.numpy()
            uniques = np.unique(labels).tolist()

        dim_l = len(uniques)

        dims = dict(images=dim_images, targets=dim_l)
        input_names = ['images', 'targets', 'index']

        self.add_dataset(
            source,
            data=dict(train=train_set, test=test_set),
            input_names=input_names,
            dims=dims,
            scale=scale
        )


register_plugin(TorchvisionDatasetPlugin)
