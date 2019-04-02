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
               tanh_normalization=False, image_size=224, **transform_args):
        '''

        Args:
            normalize: Normalize imagenet.
            tanh_normalization: Scale data from -1 to 1.
            **transform_args: Extra transformation arguments.

        Returns:

        '''

        Dataset = self.make_indexing(torchvision.datasets.ImageFolder)
        data_path = self.get_path(source)

        if isinstance(data_path, dict):
            if 'train' not in data_path.keys() and 'valid' in data_path.keys():
                raise ValueError('Imagenet data path must have `train` and '
                                 '`valid` paths specified')
            train_path = data_path['train']
            test_path = data_path['valid']
        else:
            train_path = path.join(data_path, 'train')
            test_path = path.join(data_path, 'val')

        if copy_to_local:
            train_path = self.copy_to_local_path(train_path)
            test_path = self.copy_to_local_path(test_path)

        if normalize and isinstance(normalize, bool):
            if tanh_normalization:
                normalize = transforms.Normalize((0.5, 0.5, 0.5),
                                                 (0.5, 0.5, 0.5))
            else:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

        if source == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
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

        dim_images = train_set[0][0].size()

        print('Computing min / max...')

        img_min = 1000
        img_max = -1000
        for i in range(1000):
            img = train_set[i][0]
            img_min = min(img.min(), img_min)
            img_max = max(img.max(), img_max)

        dim_l = len(train_set.classes)

        dims = dict(images=dim_images, targets=dim_l)

        self.add_dataset(
            source,
            data=dict(train=train_set, test=test_set),
            input_names=input_names,
            dims=dims,
            scale=(img_min, img_max))


register_data(ImageFolder)
