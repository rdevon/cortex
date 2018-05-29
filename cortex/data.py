"""
TODO
"""
from cortex.models import Sobel

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
import shutil
from os import path
import torch
import torchvision.transforms as transforms

LOGGER = logging.getLogger('cortex.data')

IMAGE_SCALE = [0, 1]

_args = dict(
    source=None,
    batch_size=64,
    n_workers=4,
    skip_last_batch=False,
    test_on_train=False,
    transform_args={},
)

_args_help = dict(
    source='Dataset (location (full path) or name).',
    batch_size='Batch size',
    n_workers='Number of workers',
    skip_last_batch='Skip the last batch of the epoch',
    test_on_train='Use train set on evaluation',
    transform_args='Transformation args for the data. Keywords: normalize (bool), center_crop (int), '
                   'image_size (int or tuple), random_crop (int), use_sobel (bool), random_resize_crop (int), '
                   'or flip (bool)',
)

CONFIG = None

def set_config(config):
    """
    TODO
    :param config:
    :type config:
    """
    global CONFIG
    CONFIG = config


def make_transform(source, normalize=True, center_crop=None, image_size=None,
                   random_crop=None, flip=None, random_resize_crop=None,
                   random_sized_crop=None, use_sobel=False):
    """
    TODO
    :param source:
    :type source:
    :param normalize:
    :type normalize:
    :param center_crop:
    :type center_crop:
    :param image_size:
    :type image_size:
    :param random_crop:
    :type random_crop:
    :param flip:
    :type flip:
    :param random_resize_crop:
    :type random_resize_crop:
    :param random_sized_crop:
    :type random_sized_crop:
    :param use_sobel:
    :type use_sobel:
    :return:
    :rtype:
    """
    default_normalization = {
        'MNIST': [(0.5,), (0.5,)],
        'dSprites': [(0.5,), (0.5,)],
        'Fashion-MNIST': [(0.5,), (0.5,)],
        'EMNIST': [(0.5,), (0.5,)],
        'PhotoTour': [(0.5,), (0.5,)],
        'Imagenet-12': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        'Tiny-Imagenet': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
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

    if random_resize_crop:
        transform_.append(
            transforms.RandomResizedCrop(
                random_resize_crop, scale=(
                    0.5, 1)))
    elif random_crop:
        transform_.append(transforms.RandomSizedCrop(random_crop))
    elif center_crop:
        transform_.append(transforms.CenterCrop(image_crop))
    elif random_sized_crop:
        transform_.append(transforms.RandomSizedCrop(random_sized_crop))

    if image_size:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_.append(transforms.Resize(image_size))

    if flip:
        if isinstance(flip, bool):
            flip = 0.5
        transform_.append(transforms.RandomHorizontalFlip())

    transform_.append(transforms.ToTensor())

    if use_sobel:
        transform_.append(Sobel())

    if normalize and isinstance(normalize, bool):
        if source in default_normalization.keys():
            normalize = default_normalization[source]
        else:
            raise ValueError(
                'Default normalization for source {} not found. Please enter custom normalization.'
                ''.format(source))

    transform_.append(transforms.Normalize(*normalize))
    if normalize[0] == (0.5, 0.5, 0.5):
        IMAGE_SCALE = [-1, 1]

    transform = transforms.Compose(transform_)
    return transform


def make_indexing(C):
    """
    TODO
    """
    class IndexingDataset(C):
        def __getitem__(self, index):
            output = super().__getitem__(index)
            return output + (index,)

    return IndexingDataset


def make_tds_random_and_split(C):
    class RandomSplitting(C):
        def __init__(self, *args, idx=None, split=.8, **kwargs):
            super().__init__(*args, **kwargs)
            self.idx = idx if idx is not None else torch.randperm(len(self))
            tensors_ = []

            for i in range(len(self.tensors)):
                if split > 0:
                    tensors_.append(
                        self.tensors[i][self.idx][:int(split * len(self))])
                else:
                    tensors_.append(
                        self.tensors[i][self.idx][int(split * len(self)) - 1:])

            self.tensors = tuple(tensors_)

    return RandomSplitting


def copy_to_local_path(from_path):
    """
    TODO
    """
    if from_path.endswith('/'):
        from_path = from_path[:-1]
    basename = path.basename(from_path)
    if not CONFIG.local_data_path:
        raise ValueError(
            '`local_data_path` not set in `config.yaml`. Set this path if you want local copying.')
    to_path = path.join(CONFIG.local_data_path, basename)
    if ((not path.exists(to_path)) and path.exists(from_path)):
        logger.info('Copying {} to {}'.format(from_path, to_path))
        if path.isdir(from_path):
            shutil.copytree(from_path, to_path)
        else:
            shutil.copy(from_path, CONFIG.local_data_path)

    return to_path


