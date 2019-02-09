"""
Extra functions for build-in datasets
"""

import torchvision.transforms as transforms


def build_transforms(normalize=None, center_crop=None, image_size=None,
                     random_crop=None, flip=None, random_resize_crop=None,
                     random_sized_crop=None):
    """

    Args:
        normalize (tuple or transforms.Normalize): Parameters for data normalization.
        center_crop (int): Size for center crop.
        image_size (int): Size for image size.
        random_crop (int): Size for image random crop.
        flip (bool): Randomly flip the data horizontally.
        random_resize_crop (int): Random resize crop the image.
        random_sized_crop (int): Random size crop of the image.

    Returns:
        Transforms

    """

    transform_ = []

    if random_resize_crop:
        transform_.append(transforms.RandomResizedCrop(random_resize_crop))
    elif random_crop:
        transform_.append(transforms.RandomCrop(random_crop))
    elif center_crop:
        transform_.append(transforms.CenterCrop(center_crop))
    elif random_sized_crop:
        transform_.append(transforms.RandomSizedCrop(random_sized_crop))

    if image_size:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        transform_.append(transforms.Resize(image_size))

    if flip:
        transform_.append(transforms.RandomHorizontalFlip())

    transform_.append(transforms.ToTensor())

    if normalize:
        if isinstance(normalize, transforms.Normalize):
            transform_.append(normalize)
        else:
            transform_.append(transforms.Normalize(*normalize))
    transform = transforms.Compose(transform_)
    return transform
