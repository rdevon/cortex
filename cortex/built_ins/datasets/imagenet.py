'''Handler for imagenet datasets.

'''

from os import path

import torchvision
from torchvision.transforms import transforms

from cortex.plugins import DatasetPlugin, register_data
from cortex.built_ins.datasets.utils import build_transforms


class ImageFolder(DatasetPlugin):
    sources = ['tiny-imagenet-200', 'imagenet']

    def handle(self, source, copy_to_local=False, normalize=True,
               full_imagenet=False, **transform_args):

        Dataset = self.make_indexing(torchvision.datasets.ImageFolder)
        data_path = self.get_path(source)

        train_path = path.join(data_path, 'train')
        test_path = path.join(data_path, 'val')

        if copy_to_local:
            train_path = self.copy_to_local_path(train_path)
            test_path = self.copy_to_local_path(test_path)

        if normalize and isinstance(normalize, bool):
            normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

        if full_imagenet:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = build_transforms(
                normalize=normalize, **transform_args)
            test_transform = build_transforms(normalize=normalize)
        train_set = Dataset(root=train_path, transform=train_transform)
        test_set = Dataset(root=test_path, transform=test_transform)
        input_names = ['images', 'targets', 'index']

        dim_c, dim_x, dim_y = train_set[1][0].size()
        dim_l = len(train_set.classes)

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l)

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((-1, 1))


register_data(ImageFolder)
