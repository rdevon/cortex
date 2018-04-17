#!/usr/bin/env python
'''CUB dataset

Credit goes to Tristan Silvia and Margaux Luck

'''

import logging
import os

import numpy
import pandas as pd
import PIL
from PIL import Image
import torch.utils.data as data

from operator import itemgetter


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


logger = logging.getLogger('cortex.cub')


def pil_loader(path):
    """PIL loader.
    Parameters
    ----------
    path : string
        Path to the image.
    Returns
    -------
    tensor
        The image.
    """
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    """Accimage loader.
    Parameters
    ----------
    path : string
        Path to the image.
    Returns
    -------
    tensor
        The image.
    """
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """Load an image.
    Parameters
    ----------
    path : string
        Path to the image.
    Returns
    -------
    tensor
        The image.
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def is_image_file(filename):
    """Verify if a file is an image.
    Parameters
    ----------
    filename : string
        Path to the image.
    Returns
    -------
    boolean
        Indicates if it is an image or not.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(_dir):
    """Record the classes and their correspondances.
    Parameters
    ----------
    _dir : string
        Root directory path.
    Returns
    -------
    classes : list
        List of the class names.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    """
    classes = [
        d for d in os.listdir(_dir) if os.path.isdir(os.path.join(_dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(_dir, class_to_idx, boxes):
    """Record the image path and their corresponding classes.
    Parameters
    ----------
    _dir : string
        Root directory path.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    indices : type
        Description.
    boxes : dict
        Dict with items (image_name, list of coordinates for the bounded box).
    Returns
    -------
    images : list
        List of (image path, class_index) tuples
    """
    images = []
    _dir = os.path.expanduser(_dir)
    for target in sorted(os.listdir(_dir)):
        # if class_to_idx[target] not in indices:
        #     continue

        d = os.path.join(_dir, target)

        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], boxes[fname[:-4]])
                    images.append(item)

    return images


class CUBStandard(data.Dataset):
    """Dataset loader for the CUB dataset.
    Parameters
    ----------
    root : string
        Root directory path.
    transform : callable, optional
        A function/transform that  takes in an PIL image and returns a
        transformed version. E.g, ``transforms.RandomCrop``
    loader : callable, optional
        A function to load an image given its path.
    split_type : string
        The name of the split in {'train', 'test', 'full'} or the name of the
        class.
    threshold_attrs : boolean
        Use a threshold or not for the attributes.
    imsize : int
        The image size.
    seed : int or str
        If str, it loads a specifit classes split from a file.
        If int, used for the random seed of the random selection of classes
        split.

    """

    def __init__(
            self, root='/data/lisa/data/ZSL/CUB_200_2011',
            transform=None,
            loader=default_loader, split_type='train',
            threshold_attrs=False, imsize=64,
            use_bboxes=True, seed=1234):

        assert(split_type in ['train', 'test', 'val'])

        # Path to images
        self.img_path = os.path.join(root, 'images')
        self.root = root
        self.bbox = self.load_bbox()
        self.imsize = imsize

        classes, class_to_idx = find_classes(self.img_path)

        imgs = make_dataset(
                self.img_path, class_to_idx,
                self.bbox)

        if isinstance(seed, int):
            # Generate a split for the train and test sets according to seed.
            rand_state = numpy.random.RandomState(seed)
            permuted_indices = list(rand_state.permutation(len(imgs)))
            # Standard split for the CUB dataset
            # References for this split:
            # http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Akata_Label-Embedding_for_Attribute-Based_2013_CVPR_paper.pdf
            # https://arxiv.org/pdf/1409.8403.pdf
            # cub_standard_split = 150
            # indices for the train and test classes
            end_tr_indices = int(0.6 * len(permuted_indices))
            end_val_indices = int(0.8 * len(permuted_indices))

            train_indices = permuted_indices[:end_tr_indices]
            val_indices = permuted_indices[end_tr_indices:end_val_indices]
            test_indices = permuted_indices[end_val_indices:]



        # elif seed == 'SS':
        #     train_classes = pd.read_csv(
        #         './datasets/cub_split/trainvalclasses.txt',
        #         header=None)
        #     train_classes = train_classes[0].tolist()
        #     train_indices = [class_to_idx[k] for k in train_classes]
        #     test_classes = pd.read_csv(
        #         './datasets/cub_split/testclasses.txt',
        #         header=None)
        #     test_classes = test_classes[0].tolist()
        #     test_indices = [class_to_idx[k] for k in test_classes]
        else:
            raise Exception(
                "({}) is an invalid seed value for CUB dataset.".format(seed))

        # self.test_indices = test_indices
        # self.train_indices = train_indices
        assert (len(set(train_indices) & set(test_indices)) == 0)
        assert (len(set(train_indices) & set(val_indices)) == 0)
        assert (len(set(val_indices) & set(test_indices)) == 0)
        # import ipdb; ipdb.set_trace()


        if split_type == "train":
        	imgs = itemgetter(*train_indices)(imgs)
        	# imgs = imgs[train_indices]
        elif split_type == "val":
        	imgs = itemgetter(*val_indices)(imgs)
        	# imgs = imgs[val_indices]
        elif split_type == "test":
        	imgs = itemgetter(*test_indices)(imgs)
        	# imgs = imgs[test_indices]
        else:
        	raise NotImplementedError("No such split.")

        # if split_type == 'train':
        #     imgs = make_dataset(
        #         self.img_path, class_to_idx, train_indices, self.bbox)

        # elif split_type == 'test':
        #     imgs = make_dataset(
        #         self.img_path, class_to_idx, test_indices, self.bbox)

        # elif split_type == 'full':
        #     imgs = make_dataset(
        #         self.img_path, class_to_idx,
        #         train_indices + test_indices,
        #         self.bbox)

        # else:
        #     imgs = make_dataset(
        #         self.img_path, class_to_idx,
        #         [split_type],
        #         self.bbox)

        if len(imgs) == 0:
            raise(RuntimeError(
                'Found 0 images in subfolders of: ' + root + '\n'))

        # check the number of images here, etc.
        self.threshold_attrs = threshold_attrs
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader
        self.seed = seed
        self.split_type = split_type
        self.compute_class_embedding()

        # If this is False bboxes will not be used. Useful when testing on the
        # data during classify
        self.use_bboxes = use_bboxes


    def __getitem__(self, index):
        """Get the items : image, target, attributes
        Parameters
        ----------
        index : int
            Index
        Returns
        -------
        img : tensor
            The image.
        target : int
            Target is class_index of the target class.
        tensor
            The attributes of the target.
        """
        path, target, box = self.imgs[index]
        if not self.use_bboxes:
            box = None
        img = self.get_img(path, box)

        return img, target, self.attrs[target, :]

    def __len__(self):
        """Get the number of image in the dataset.
        Returns
        -------
        int
            The number of images in the dataset.
        """
        return len(self.imgs)

    def compute_class_embedding(self):
        """Word embedding.
        """
        file_path = os.path.join(
            self.root, 'attributes',
            'class_attribute_labels_continuous.txt')
        in_file = open(file_path, 'r')

        self.attrs = []
        for line in in_file:
            line = list(map(float, line.split()))
            self.attrs.append(line)

        self.attrs = numpy.array(self.attrs, dtype='float32')

        # Simply normalizing by 100
        self.attrs /= 100

        if self.threshold_attrs:
            self.attrs = numpy.where(
                self.attrs > 0.5, 1., 0.).astype('float32')

        in_file.close()

    def get_class_embedding(self):
        """Get the attributes.
        Returns
        -------
        tensor
            The attributes of the target.
        """
        return self.attrs

    # def get_test_classes(self):
    #     """Get the test classes.
    #     Returns
    #     -------
    #     list
    #         List of test classes.
    #     """
    #     return self.test_indices

    # def get_train_classes(self):
    #     """Get the train classes.
    #     Returns
    #     -------
    #     list
    #         List of the train classes.
    #     """
    #     return self.train_indices

    def get_img(self, img_path, bbox):
        """Get the image.
        Parameters
        ----------
        img_path : string
            Path to image.
        bbox : list
            List of coordinates for the bounded box.
        Returns
        -------
        img : tensor
            The image.
        """
        img = self.loader(img_path)

        width, height = img.size
        if bbox is not None:
            R = int(numpy.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = numpy.maximum(0, center_y - R)
            y2 = numpy.minimum(height, center_y + R)
            x1 = numpy.maximum(0, center_x - R)
            x2 = numpy.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])

        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def load_bbox(self):
        """Get the bounded box.
        Returns
        -------
        filename_bbox
            Dict with items (image_name, list of coordinates for the bounded
            box).
        """
        bbox_path = os.path.join(self.root, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(
            bbox_path, delim_whitespace=True, header=None).astype(int)

        filepath = os.path.join(self.root, 'images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()


        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = os.path.split(filenames[i])[-1][:-4]
            filename_bbox[key] = bbox

        return filename_bbox


if __name__ == "__main__":
	train_set = CUBStandard(split_type="train")
	test_set = CUBStandard(split_type="test")
	val_set = CUBStandard(split_type="val")

	# import ipdb; ipdb.set_trace()

	# Counting classes
	classes = []
	for i in range(len(train_set.imgs)):
		classes.append(train_set.imgs[i][1])
	print(len(set(classes)))