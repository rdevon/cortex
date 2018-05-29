import errno

from cortex import data
import os
import torch
import itertools as it

class SmallDataset(data.TensorDataset):

    def __init__(self, root, *select,
                 stardardize=False, load=False, download=False):
        """Download or load a small dataset.

        Parameters
        ----------
        root : str
           Names the path to a directory in which the dataset is or will be
           located
        select : positional arguments
           Some extra, possibly unnecessary arguments, to specify a particular
           dataset from a family of datasets to be loaded.

           What it can be, depends on a particular dataset, **please refer** to
           its documentation.
        stardardize : bool, optional, default=False
           If True, perform small preprocessing of the datasets: ``(X - mean)/std``,
           when preparing the dataset for loading
        load : bool, optional, default=False
           If True, load the dataset from local directory `root`
        download : bool, optional, default=False
           If True, download the dataset to local directory `root`
           from an online source

        """
        self.root = os.path.expanduser(root)
        self.stardardize = stardardize

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if load:
            data, target = self.prepare(*select)
            super().__init__(data, target)

    def download(self):
        """Download, and unzip in the correct location."""
        import urllib
        import zipfile

        if self.check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            ext = os.path.splitext(file_path)[1]
            with open(file_path, 'wb') as f:
                f.write(data.read())
            if ext == '.zip':
                with zipfile.ZipFile(file_path) as zip_f:
                    zip_f.extractall(self.root)
                os.unlink(file_path)

        print('Done!')

    def prepare(self, *select):
        """Make torch Tensors from data and label files."""
        datafile, labelfile = self.files(*select)
        data_filepath = os.path.join(self.root, datafile)
        label_filepath = os.path.join(self.root, labelfile)
        data = []
        target = []
        with open(data_filepath) as data_f, open(label_filepath) as label_f:
            for x, y in zip(data_f, it.islice(label_f, self.sync_files, None)):
                data.append(list(map(int, x.split())))
                target.append(int(y))
        data = torch.Tensor(data)
        target = torch.Tensor(target)

        if self.stardardize:
            data_mean = data.mean(dim=0, keepdim=True)
            data_std = data.std(dim=0, keepdim=True)
            data = (data - data_mean) / data_std

        return data, target

    def files(self, *select):
        return '', ''

    def check_exists(self):
        return True