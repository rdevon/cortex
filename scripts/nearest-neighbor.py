from sklearn.neighbors import NearestNeighbors
from PIL import Image
import numpy as np
import os
from operator import mul
import functools


def get_neighbors(samples, dataset, n_neighbors):
    size = functools.reduce(mul, dataset[0].shape, 1)
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric='l2', algorithm='brute').fit(
        dataset.reshape(-1, size))
    _, samples_idxs = nbrs.kneighbors(samples.reshape(-1, size))
    return np.array([[dataset[idx] for idx in idxs] for idxs in samples_idxs])


if __name__ == '__main__':
    import torchvision
    import torch
    root = os.argv[1]
    images = np.concatenate([
        np.array([np.array(Image.open(os.path.join(
            root, 'n01443537', 'images', file)))
            for file in os.listdir(os.path.join(
                root, 'n01443537', 'images'))]) / 255.,
        np.array([np.array(Image.open(os.path.join(
            root, 'n09193705', 'images', file)))
            for file in os.listdir(os.path.join(
                root, 'n09193705', 'images'))]) / 255.,
        np.array([np.array(Image.open(os.path.join(
            root, 'n01742172', 'images', file)))
            for file in os.listdir(os.path.join(
                root, 'n01742172', 'images'))]) / 255.,
        np.array([np.array(Image.open(os.path.join(
            root, 'n02058221', 'images', file)))
            for file in os.listdir(os.path.join(
                root, 'n02058221', 'images'))]) / 255.,
        np.array([np.array(Image.open(os.path.join(
            root, 'n02094433', 'images', file)))
            for file in os.listdir(os.path.join(
                root, 'n02094433', 'images'))]) / 255.],
        axis=0)
    nbhrs = np.concatenate((get_neighbors(images[[0, -1]], images, 5)), axis=0)
    torchvision.utils.save_image(torch.from_numpy(nbhrs.transpose(0, 3, 1, 2)),
                                 'test.png', nrow=5)
