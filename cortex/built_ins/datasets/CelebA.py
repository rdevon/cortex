'''Handler for CelebA.

'''

import os

import torchvision

from cortex.plugins import DatasetPlugin, register_plugin
from cortex.built_ins.datasets.utils import build_transforms


class CelebAPlugin(DatasetPlugin):
    sources = ['CelebA']

    def handle(self, source, copy_to_local=False, normalize=True,
               **transform_args):
        Dataset = self.make_indexing(CelebA)
        data_path = self.get_path(source)

        if copy_to_local:
            data_path = self.copy_to_local_path(data_path)

        if normalize and isinstance(normalize, bool):
            normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

        transform = build_transforms(normalize=normalize, **transform_args)

        train_set = Dataset(root=data_path, transform=transform, download=True)
        test_set = Dataset(root=data_path, transform=transform)
        input_names = ['images', 'targets']

        dim_c, dim_x, dim_y = train_set[0][0].size()
        dim_l = len(train_set.classes)

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l)

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((-1, 1))


register_plugin(CelebAPlugin)


class CelebA(torchvision.datasets.ImageFolder):
    url = ('https://www.dropbox.com/sh/8oqt9vytwxb3s4r/'
           'AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1')
    filename = "img_align_celeba.zip"

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            download=False):
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
                print('Dataset exists, not downloading.')
                return
            else:
                raise

        # downloads file
        if os.path.isfile(fpath):
            print('Using downloaded file: {}'.format(fpath))
        else:
            try:
                print('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    print('Failed download. Trying https -> http instead.'
                          ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(url, fpath)
                else:
                    raise

        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(image_dir)
        zip_ref.close()
