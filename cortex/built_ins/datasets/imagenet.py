'''Handler for imagenet datasets.

'''

import torchvision

from cortex.plugins import DatasetPlugin, register_data
from cortex.built_ins.datasets.utils import build_transforms


class ImageFolder(DatasetPlugin):
    sources = ['tiny-imagenet-200']

    def handle(self, source, copy_to_local=False, normalize=True,
               **transform_args):

        Dataset = self.make_indexing(torchvision.datasets.ImageFolder)
        data_path = self.get_path(source)

        if isinstance(data_path, dict):
            train_path = data_path['train']
            test_path = data_path['test']
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
                test_path = self.copy_to_local_path(test_path)
        elif isinstance(data_path, (tuple, list)):
            train_path, test_path = data_path
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
                test_path = self.copy_to_local_path(test_path)
        else:
            train_path = data_path
            if copy_to_local:
                train_path = self.copy_to_local_path(train_path)
            test_path = data_path

        if normalize and isinstance(normalize, bool):
            normalize = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]

        transform = build_transforms(normalize=normalize, **transform_args)

        train_set = Dataset(root=train_path, transform=transform)
        test_set = Dataset(root=test_path, transform=transform)
        input_names = ['images', 'targets']

        dim_c, dim_x, dim_y = train_set[0][0].size()
        dim_l = len(train_set.classes)

        dims = dict(x=dim_x, y=dim_y, c=dim_c, labels=dim_l)

        self.add_dataset('train', train_set)
        self.add_dataset('test', test_set)
        self.set_input_names(input_names)
        self.set_dims(**dims)

        self.set_scale((-1, 1))


register_data(ImageFolder)
