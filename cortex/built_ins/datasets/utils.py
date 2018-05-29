'''

'''

import torchvision.transforms as transforms


def build_transform(normalize=True, center_crop=None, image_size=None,
                    random_crop=None, flip=None, random_resize_crop=None,
                    random_sized_crop=None, use_sobel=False):
    global IMAGE_SCALE
    transform_ = []

    if random_resize_crop:
        transform_.append(transforms.RandomResizedCrop(random_resize_crop, scale=(0.5, 1)))
    elif random_crop:
        transform_.append(transforms.RandomSizedCrop(random_crop))
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

    if use_sobel:
        transform_.append(Sobel())

    transform_.append(transforms.Normalize(*normalize))
    if normalize[0] == (0.5, 0.5, 0.5):
        IMAGE_SCALE = [-1, 1]

    transform = transforms.Compose(transform_)
    return transform