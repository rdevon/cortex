"""Data module

This module defines the main iteration functionality.

"""

import signal

import torch
from progressbar import Bar, ProgressBar, Percentage, Timer, ETA

from .noise import get_noise_var
from .. import exp

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


class DataHandler:
    """Handler for torchvision datasets.

    This data handler is specialized for torchvision datasets.

    """
    def __init__(self):
        self.dims = {}
        self.input_names = {}
        self.noise = {}
        self.loaders = {}
        self.datasets = {}
        self.batch = None
        self.noise = {}
        self.iterator = {}
        self.pbar = None
        self.u = 0
        self.mode = None
        self.inputs = dict()

    def set_batch_size(self, batch_size, skip_last_batch=False):
        """Sets the batch sizes for the data handler.

        Args:
            batch_size: The batch size.
            skip_last_batch: Whether to skip the last batch in case
            the number of examples is different.

        """

        self.batch_size = batch_size
        self.skip_last_batch = skip_last_batch

    def set_input_names(self, **kwargs):
        """Sets input names to allow for variable tensor names.

        Args:
            **kwargs: dictionary of name maps (name from: name to)

        """
        self.inputs.update(**kwargs)

    def add_dataset(self, datasets, source, name, plugin,
                    n_workers=4, shuffle=True, DataLoader=None):
        """Adds a dataset to the data handler.

        Args:
            datasets: dictionary of datasets added to cortex.
            source: Name of the dataset.
            plugin: Plugin for the dataset. This will handle dataset creation.
            n_workers: Number of workers for iterator.
            shuffle: Shuffle the dataset for every epoch.
            DataLoader: Specialized data loader (user-defined).

        """

        DataLoader = (DataLoader or plugin._dataloader_class or
                      torch.utils.data.DataLoader)
        data = datasets[source]['data']
        dims = datasets[source]['dims']
        input_names = datasets[source]['input_names']

        if len(data) == 0:
            raise ValueError('No datasets found in plugin')

        loaders = {}
        for k, dataset in data.items():
            N = len(dataset)
            dims['N_' + k] = N

            if isinstance(self.batch_size, dict):
                try:
                    self.batch_size[k]
                except KeyError:
                    self.batch_size[k] = self.batch_size_
                finally:
                    batch_size = self.batch_size[k]
            else:
                self.batch_size_ = self.batch_size
                self.batch_size = {k: self.batch_size_}
                batch_size = self.batch_size_

            loaders[k] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=shuffle, num_workers=n_workers,
                                    worker_init_fn=lambda x:
                                    signal.signal(signal.SIGINT,
                                                  signal.SIG_IGN),
                                    pin_memory=True)

        self.datasets[name] = dataset
        self.dims[name] = dims
        self.input_names[name] = input_names
        self.loaders[name] = loaders

    def add_noise(self, key, dist=None, size=None, **kwargs):
        """Adds a noise variable to the data handler.

        Given a specified noise distribution, dimensionalities, and other noise parameters,
        this will add a noise variable that can be accessed within a model.

        Args:
            key: Name of noise variable, e.g., `Z`.
            dist: Distribution of noise variable, e.g., `Normal` or `Uniform`.
            size: Size of the tensor drawn from noise.
            **kwargs: Ductionary of parameters specific for distribution.

        """

        if size is None:
            raise ValueError

        dim = size

        if not isinstance(size, tuple):
            size = (size,)

        train_size = (self.batch_size['train'],) + size
        test_size = (self.batch_size['test'],) + size

        var = get_noise_var(dist, train_size, **kwargs)
        var_t = get_noise_var(dist, test_size, **kwargs)

        self.noise[key] = dict(train=var, test=var_t)
        self.dims[key] = dim

    def __iter__(self):
        """__iter__

        Returns:
            self

        """
        return self

    def __next__(self):
        """__next__ function that returns next batch.

        Loops through the sources (as defined by loaders) and draws batches from
        each iterator.

        For each source, we first:
            1) Draw a tuple of data from the iterator
            2) Check if the number of examples is consistent with other sources.
            3) Form a dictionary of input names for the source and the data tuple.
            4) Add this dictionary to a dictionary of outputs.
            5) Add any noise variables.

        Returns:
            Tuple of batches.

        """
        output = {}
        sources = self.loaders.keys()

        batch_size = self.batch_size[self.mode]
        for source in sources:
            data = next(self.iterators[source])
            if data[0].size()[0] < batch_size:
                if self.skip_last_batch:
                    raise StopIteration
                batch_size = data[0].size()[0]
            data = dict((k, v) for k, v in zip(self.input_names[source], data))

            output[source] = data

        for k, n_vars in self.noise.items():
            n_var = n_vars[self.mode]
            n_var = n_var.sample()
            n_var = n_var.to(exp.DEVICE)

            if n_var.size()[0] != batch_size:
                n_var = n_var[0:batch_size]
            output[k] = n_var

        self.batch = output
        self.u += 1
        self.update_pbar()

        if self.mode == 'train':
            exp.INFO['data_steps'] += 1

        return self.batch

    def next(self):
        return self.__next__()

    def __getitem__(self, key):
        """Returns tensor from batch with key.

        If the key has a `.` in it, this indicates to traverse the batch dictionary
        to the next level.

        Returns:
            Tensor of data.

        """
        if self.batch is None:
            raise RuntimeError('Batch not set')

        def get_data_keys(d):
            keys = []
            for k, v in d.items():
                keys.append(k)
                if isinstance(v, dict):
                    keys_ = get_data_keys(v)
                    for k_ in keys_:
                        keys.append(k + '.' + k_)
            return keys

        def get_data(d, key):
            if '.' in key:
                k_ = key.split('.')
                head = k_[0]
                key = '.'.join(k_[1:])
            elif len(d.keys()) == 1:
                head = list(d.keys())[0]
            else:
                head = None

            if head:
                return get_data(d[head], key)

            try:
                key = self.inputs.get(key, key)
                return d[key]
            except KeyError:
                raise KeyError('Data with label `{}` not found. Available: {}'
                               .format(key, get_data_keys(d)))

        return get_data(self.batch, key)

    def get_batch(self, *keys):
        """Retruns a batch of multiple inputs.

        Args:
            *keys: List of keys.

        Returns:
            List of batches.

        """
        if self.batch is None:
            raise RuntimeError('Batch not set')

        batch = []
        for k in keys:
            b = self[k]
            batch.append(b)
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def get_dims(self, q):
        """Returns the dimensionality of input.

        This is useful for network formation, e.g., where the initial layers and
        dependent on the input size.

        Args:
            q: query or list of queries.

        Returns:
            List of dimensions.

        """
        if not isinstance(q, list):
            q = [q]

        dims = []
        for q_ in q:
            if '.' in q_:
                head, tail = q_.split('.')
            elif len(self.dims.keys()) == 1:
                head = list(self.dims.keys())[0]
                tail = q_
            else:
                raise KeyError('Error with dimension query {}. '
                               'Available: {}'.format(q_, self.dims))
            try:
                dims.append(self.dims[head][tail])
            except KeyError:
                raise KeyError('Dimensions {} not found. '
                               'Available from data handler: {}'.format(q_, self.dims))

        if len(dims) == 1:
            return dims[0]
        else:
            return dims

    def get_label_names(self, source=None):
        # TODO(Devon): This needs to
        # incorporate specific label
        # names from the dataset plugin.
        source = source or list(self.loaders.keys())[0]
        names = ['{}'.format(i) for i in range(self.dims[source]['targets'])]
        return names

    def make_iterator(self, source):
        loader = self.loaders[source][self.mode]

        def iterator():
            for inputs in loader:
                inputs = [inp.to(exp.DEVICE) for inp in inputs]
                inputs_ = []
                for i, inp in enumerate(inputs):
                    inputs_.append(inp)
                yield inputs_
        return iterator()

    def update_pbar(self):
        """Updates the progress bar for the command line.

        """
        if self.pbar:
            self.pbar.update(self.u)

    def reset(self, mode, make_pbar=True, string=''):
        """Resets the data iterator.

        Args:
            mode: Which mode to reset to, e.g., `train` or `test`.
            make_pbar: Whether to use a pbar in the next epoch.
            string: String to use in the pbar.

        """
        self.mode = mode
        self.u = 0

        if make_pbar:
            widgets = [string, Timer(), ' | ',
                       Percentage(), ' | ', ETA(), Bar()]
            if len([len(loader[self.mode]) for loader
                    in self.loaders.values()]) == 0:
                maxval = 1000
            else:
                maxval = min(len(loader[self.mode])
                             for loader in self.loaders.values())
            self.pbar = ProgressBar(widgets=widgets, maxval=maxval).start()
        else:
            self.pbar = None

        sources = self.loaders.keys()
        self.iterators = dict((source, self.make_iterator(source))
                              for source in sources)
