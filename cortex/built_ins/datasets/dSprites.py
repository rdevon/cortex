'''dShapes dataset

Taken and adapted from https://github.com/Near32/PYTORCH_VAE

'''


from os import path
import urllib.request

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from . import logger


DATASETS = ['dSprites']


class dSprites(Dataset):
    _url = ('https://github.com/deepmind/dsprites-dataset/blob/master/'
            'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true')

    def __init__(self, root, download=True, transform=None, shuffle=False):
        if not root:
            raise ValueError('Dataset path not provided')
        self.root = root
        self.transform = transform

        if download:
            if path.isfile(root):
                logger.warning('File already in path, ignoring download.')
            else:
                urllib.request.urlretrieve(self._url, root)

        # Load dataset
        dataset_zip = np.load(self.root)
        logger.debug('Keys in the dataset:', dataset_zip.keys())
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        logger.info('Dataset loaded : OK.')

        if shuffle:
            self.idx = np.random.permutation(len(self))
            self.imgs = self.imgs[self.idx]
            self.latents_classes = self.latents_classes[self.idx]
            self.latents_values = self.latents_values[self.idx]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = Image.fromarray(self.imgs[idx])
        latent = self.latents_values[idx]

        if self.transform is not None:
            image = self.transform(image)

        sample = (image, latent)

        return sample
