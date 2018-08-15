"""
Extra functions for build-in datasets
"""

import torchvision.transforms as transforms


def build_transforms(normalize=True, center_crop=None, image_size=None,
                     random_crop=None, flip=None, random_resize_crop=None,
                     random_sized_crop=None, use_sobel=False):
    """

    Args:
        normalize:
        center_crop:
        image_size:
        random_crop:
        flip:
        random_resize_crop:
        random_sized_crop:
        use_sobel:

    Returns:

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
