'''

'''

from os import path
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from . import logger, CONFIG


class Sobel(object):
    def __init__(self):
        self.kernel_g_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_g_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)

    def _apply_sobel(self, channel):
        g_x = F.conv2d(channel, self.kernel_g_x, stride=1, padding=1)
        g_y = F.conv2d(channel, self.kernel_g_y, stride=1, padding=1)
        return torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))

    def __call__(self, img):
        a = torch.cat([self._apply_sobel(channel.unsqueeze(0).unsqueeze(0)) for channel in img]).squeeze(1)
        return a

    def __repr__(self):
        return self.__class__.__name__ + '()'


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