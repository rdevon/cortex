"""Data module"""

import signal

import torch
from progressbar import Bar, ProgressBar, Percentage, Timer, ETA

from .noise import get_noise_var
from .. import exp

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'


class DataHandler:
    def __init__(self):
        self.dims = {}
        self.input_names = {}
        self.noise = {}
        self.loaders = {}
        self.batch = None
        self.noise = {}
        self.iterator = {}
        self.pbar = None
        self.u = 0
        self.inputs = dict()

    def set_batch_size(self, batch_size, skip_last_batch=False):
        self.batch_size = batch_size
        self.skip_last_batch = skip_last_batch

    def set_inputs(self, **kwargs):
        self.inputs.update(**kwargs)

    def add_dataset(self, source, dataset_entrypoint,
                    n_workers=4, shuffle=True, DataLoader=None):
        DataLoader = (DataLoader or dataset_entrypoint._dataloader_class or
                      torch.utils.data.DataLoader)

        if len(dataset_entrypoint._datasets) == 0:
            raise ValueError('No datasets found in entrypoint')

        loaders = {}
        for k, dataset in dataset_entrypoint._datasets.items():
            N = len(dataset)
            dataset_entrypoint._dims['N_' + k] = N

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
                                                  signal.SIG_IGN))

        self.dims[source] = dataset_entrypoint._dims
        self.input_names[source] = dataset_entrypoint._input_names
        self.loaders[source] = loaders

    def add_noise(self, key, dist=None, size=None, **kwargs):
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
        return self

    def __next__(self):
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
            if len(sources) > 1:
                output[source] = data
            else:
                output.update(**data)

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

        return self.batch

    def next(self):
        return self.__next__()

    def __getitem__(self, item):
        if self.batch is None:
            raise RuntimeError('Batch not set')

        item = self.inputs.get(item, item)
        if item not in self.batch.keys():
            raise KeyError('Data with label `{}` not found. Available: {}'
                           .format(item, tuple(self.batch.keys())))
        batch = self.batch[item]

        return batch

    def get_batch(self, *item):
        if self.batch is None:
            raise RuntimeError('Batch not set')

        batch = []
        for i in item:
            if '.' in i:
                j, i_ = i.split('.')
                j = int(j)
                batch.append(self.batch[list(self.batch.keys())[j - 1]][i_])
            elif i not in self.batch.keys():
                raise KeyError('Data with label `{}` not found. Available: {}'
                               .format(i, tuple(self.batch.keys())))
            else:
                batch.append(self.batch[i])
        if len(batch) == 1:
            return batch[0]
        else:
            return batch

    def get_dims(self, *q):
        if q[0] in self.dims.keys():
            dims = self.dims
        else:
            key = [k for k in self.dims.keys()
                   if k not in self.noise.keys()][0]
            dims = self.dims[key]

        try:
            d = [dims[q_] for q_ in q]
        except KeyError:
            raise KeyError('Cannot resolve dimensions {}, provided {}'
                           .format(q, dims))
        if len(d) == 1:
            return d[0]
        else:
            return d

    def get_label_names(self, source=None):
        # TODO(Devon): This needs to
        # incorporate specific label
        # names from the dataset plugin.
        source = source or list(self.loaders.keys())[0]
        names = ['{}'.format(i) for i in range(self.dims[source]['labels'])]
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
        if self.pbar:
            self.pbar.update(self.u)

    def reset(self, mode, make_pbar=True, string=''):
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
