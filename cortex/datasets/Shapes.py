from cortex.datasets import SmallDataset
import torch
import os

class Shapes(SmallDataset):
    """Wrap shapes datasets from the website."""

    def prepare(self):
        """Make torch Tensors from data and label files."""
        datafile = self.urls[0].rpartition('/')[2]
        data_filepath = os.path.join(self.root, datafile)
        data = []
        target = []
        with open(data_filepath) as data_f:
            for sample in data_f:
                x, y, label = tuple(map(float, sample.split()))
                data.append([x, y])
                target.append(int(label) - 1)
        data = torch.Tensor(data)
        target = torch.Tensor(target)

        if self.stardardize:
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            data = (data - data_mean) / data_std

        return data, target

    def check_exists(self):
        datafile = self.urls[0].rpartition('/')[2]
        return os.path.exists(os.path.join(self.root, datafile))
