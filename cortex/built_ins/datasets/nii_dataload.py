'''Module for handling neuroimaging data
 We build an "ImageFolder" object and we can iterate/index through
 it.  The class is initialized with a folder location, a loader (the only one
    we have now is for nii files), and (optionally) a list of regex patterns.

 The user can also provide a 3D binary mask (same size as data) to vectorize
 the space/voxel dimension. Can handle 3D and 3D+time (4D) datasets So, it can
 be built one of two ways:
 1: a path to one directory with many images, and the classes are based on
 regex patterns.
   example 1a: "/home/user/some_data_path" has files *_H_*.nii and *_S_*.nii
               files
  patterned_images = ImageFolder("/home/user/some_data_path",
                     patterns=['*_H_*','*_S_*'] , loader=nii_loader)
    example 1b: "/home/user/some_data_path" has files *_H_*.nii and *_S_*.nii
    files, and user specifies a mask to vectorize space
  patterned_images_mask = ImageFolder("/home/user/some_data_path",
    patterns=['*_H_*','*_S_*'] , loader=nii_loader,
              mask="/home/user/maskImage.nii")

 2: a path to a top level directory with sub directories denoting the classes.
    example 2a: "/home/user/some_data_path" has subfolders 0 and 1 with nifti
    files corresponding to class 0 and class 1 respectively
  foldered_images = ImageFolder("/home/user/some_data_path",loader=nii_loader)
    example 2b: Same as above but with a mask
  foldered_images = ImageFolder("/home/user/some_data_path",loader=nii_loader,
                                mask="/home/user/maskImage.nii")


 The final output (when we call __getitem__) is a tuple of: (image,label)
'''

import torch.utils.data as data

import os
import os.path
import numpy as np
import nibabel as nib
from glob import glob

IMG_EXTENSIONS = ['.nii', '.nii.gz', '.img', '.hdr', '.img.gz', '.hdr.gz']


def make_dataset(dir, patterns=None):
    """

    Args:
        dir:
        patterns:

    Returns:

    """
    images = []

    dir = os.path.expanduser(dir)

    file_list = []

    all_items = [os.path.join(dir, i) for i in os.listdir(dir)]
    directories = [os.path.join(dir, d) for d in all_items if os.path.isdir(d)]
    if patterns is not None:
        for i, pattern in enumerate(patterns):
            files = [(f, i) for f in glob(os.path.join(dir, pattern))]
            file_list.append(files)
    else:
        file_list = [[(os.path.join(p, f), i)
                      for f in os.listdir(p)
                      if os.path.isfile(os.path.join(p, f))]
                     for i, p in enumerate(directories)]

    for i, target in enumerate(file_list):
        for item in target:
            images.append(item)

    return images


def nii_loader(path):
    """

    Args:
        path:

    Returns:

    """
    img = nib.load(path)
    data = img.get_data()
    # hdr = img.header

    return data


class ImageFolder(data.Dataset):
    '''
    Args:
        root (string): Root directory path.
        patterns (list): list of regex patterns
        loader (callable, optional): A function to load an image given its
        path.

     Attributes:
        imgs (list): List of (image path, class_index) tuples
    '''

    def __init__(self, root, loader=nii_loader, patterns=None, mask=None):
        imgs = make_dataset(root, patterns)

        if len(imgs) == 0:
            raise (
                RuntimeError(
                    "Found 0 images in subfolders of: " +
                    root +
                    "\n"
                    "Supported image extensions are: " +
                    ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs

        self.loader = loader
        self.mask = mask

    def maskData(self, data):
        """

        Args:
            data:

        Returns:

        """

        msk = nib.load(self.mask)
        mskD = msk.get_data()
        if not np.all(np.bitwise_or(mskD == 0, mskD == 1)):
            raise ValueError("Mask has incorrect values.")
        # nVox = np.sum(mskD.flatten())
        if data.shape[0:3] != mskD.shape:
            raise ValueError((data.shape, mskD.shape))

        msk_f = mskD.flatten()
        msk_idx = np.where(msk_f == 1)[0]

        if len(data.shape) == 3:
            data_masked = data.flatten()[msk_idx]

        if len(data.shape) == 4:
            data = np.transpose(data, (3, 0, 1, 2))
            data_masked = np.zeros((data.shape[0], int(mskD.sum())))
            for i, x in enumerate(data):
                data_masked[i] = x.flatten()[msk_idx]

        img = data_masked

        return np.array(img)

    '''
        Gives us a tuple from the array at (index) of: (image, label)
    '''

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target
            class.
        """
        path, label = self.imgs[index]
        img = self.loader(path)
        if self.mask:
            img = self.maskData(img)

        return np.array(img), label

    def __len__(self):
        return len(self.imgs)
