"""
Handler for CelebA.
"""

import csv
import os

import numpy as np
import torchvision
from torchvision.transforms import transforms

from cortex.plugins import DatasetPlugin, register_plugin
from cortex.built_ins.datasets.utils import build_transforms


class CelebAPlugin(DatasetPlugin):
    sources = ['CelebA']

    def handle(self, source, copy_to_local=False, normalize=True,
               split=None, classification_mode=False, **transform_args):
        """

        Args:
            source:
            copy_to_local:
            normalize:
            **transform_args:

        Returns:

        """
        Dataset = self.make_indexing(CelebA)
        data_path = self.get_path(source)

        if copy_to_local:
            data_path = self.copy_to_local_path(data_path)

        if normalize and isinstance(normalize, bool):
            normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

        if classification_mode:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize),
            ])
            test_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(*normalize),
            ])
        else:
            train_transform = build_transforms(normalize=normalize,
                                               **transform_args)
            test_transform = train_transform

        if split is None:
            train_set = Dataset(root=data_path, transform=train_transform,
                                download=True)
            test_set = Dataset(root=data_path, transform=test_transform)
        else:
            train_set, test_set = self.make_split(
                data_path, split, Dataset, train_transform, test_transform)
        input_names = ['images', 'labels', 'attributes']

        dim_c, dim_x, dim_y = train_set[0][0].size()
        dim_l = len(train_set.classes)
        dim_a = train_set.attributes[0].shape[0]

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l, attributes=dim_a)
        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((-1, 1))

    def make_split(self, data_path, split, Dataset, train_transform,
                   test_transform):
        train_set = Dataset(root=data_path, transform=train_transform,
                            download=True, split=split)
        test_set = Dataset(root=data_path, transform=test_transform,
                           split=split - 1)
        return train_set, test_set


register_plugin(CelebAPlugin)


class CelebA(torchvision.datasets.ImageFolder):

    url = ('https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
           'AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1')
    attr_url = ('https://www.dropbox.com/s/auexdy98c6g7y25/'
                'list_attr_celeba.zip?dl=1')
    filename = 'img_align_celeba.zip'
    attr_filename = 'list_attr_celeba.zip'

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            download=False,
            split=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.attributes = []

        attr_fpath = os.path.join(root, 'attributes', 'list_attr_celeba.txt')
        reader = csv.reader(open(attr_fpath), delimiter=' ',
                            skipinitialspace=True)
        for i, line in enumerate(reader):
            if i == 0:
                pass
            elif i == 1:
                self.attribute_names = line
            else:
                attr = ((np.array(line[1:]).astype('int8') + 1) / 2).astype('float32')
                self.attributes.append(attr)

        super(CelebA, self).__init__(root, transform, target_transform)
        if split:
            if split > 0:
                index = int(split * len(self))
                self.imgs = self.imgs[:index]
                self.attributes = self.attributes[:index]
                self.samples = self.samples[:index]
            else:
                index = int(split * len(self)) - 1
                self.imgs = self.imgs[index:]
                self.attributes = self.attributes[index:]
                self.samples = self.samples[index:]

    def __len__(self):
        return len(self.imgs)

    def download(self):
        """

        Returns:

        """
        import errno
        import zipfile
        from six.moves import urllib

        root = self.root
        url = self.url
        attr_url = self.attr_url

        root = os.path.expanduser(root)
        fpath = os.path.join(root, self.filename)
        attr_fpath = os.path.join(root, self.attr_filename)
        image_dir = os.path.join(root, 'images')
        attribute_dir = os.path.join(root, 'attributes')

        def get_data(data_path, zip_path, url):
            try:
                os.makedirs(data_path)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    return
                else:
                    raise

            if os.path.isfile(zip_path):
                print('Using downloaded file: {}'.format(zip_path))
            else:
                try:
                    print('Downloading ' + url + ' to ' + zip_path)
                    urllib.request.urlretrieve(url, zip_path)
                except Exception:
                    if url[:5] == 'https':
                        url = url.replace('https:', 'http:')
                        print('Failed download. Trying https -> http instead.'
                              ' Downloading ' + url + ' to ' + zip_path)
                        urllib.request.urlretrieve(url, zip_path)
                    else:
                        raise
            print('Unzipping {}'.format(zip_path))

            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(data_path)
            zip_ref.close()

        get_data(image_dir, fpath, url)
        get_data(attribute_dir, attr_fpath, attr_url)

    def __getitem__(self, index):
        output = super().__getitem__(index)
        return output + (self.attributes[index],)
