'''Handlers.

'''

from collections.abc import MutableMapping
import logging

import torch

logger = logging.getLogger('cortex.handlers')


class Handler(MutableMapping):
    _type = None
    _get_error_string = 'Attribute `{}` not found. Available: {}'

    def __init__(self, allow_overwrite=True, **kwargs):
        self._allow_overwrite = allow_overwrite
        self._locked = False
        self._storage = dict(**kwargs)

    def _check_type(self, value):
        if self._type and not isinstance(value, self._type):
            raise TypeError('Invalid type ({}), expected {}.'
                            .format(type(value), self._type))

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except:
            raise AttributeError(self._get_error_string
                                 .format(key, tuple(self.__dict__.keys())))

    def __delitem__(self, key):
        del self.__dict__[key]

    def __setitem__(self, key, value):
        self._check_type(value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        self._check_type(value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')

        super().__setattr__(key, value)

    def __iter__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return iter(d)

    def __len__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return len(d)

    def lock(self):
        self._locked = True


def convert_nested_dict_to_handler(d, _class=Handler):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = convert_nested_dict_to_handler(v)

    return _class(**d)


class CallSetterHandler(Handler):
    '''A handler that calls a callable when set.

    '''

    def __init__(self, fn):
        if not callable(fn):
            raise ValueError('{} is not callable.'.format(fn))
        self._fn = fn

    def __setitem__(self, key, value):
        self._fn(key, value)
        super().__setitem__(key, value)


class Alias():
    '''An alias class for referencing objects in a base container.

    '''

    def __init__(self, data, key):
        self._data = data
        self._key = key
        self._isset = False

    @property
    def value(self):
        if isinstance(self._key, (tuple, list)):
            return tuple(self._data[k] for k in self._key)

        if self._key in self._data:
            self._isset = True
            return self._data[self._key]
        else:
            self._isset = False
            return None

    @value.setter
    def value(self, value):
        self._isset = True
        if self._key in self._data and self._data[self._key] is not None:
            raise RuntimeError('Alias cannot overwrite data.')
        self._data[self._key] = value
        self._isset = True

    @property
    def isset(self):
        return self._isset


class AliasHandler(Handler):
    '''A handler for aliasing objects.

    '''
    _type = Alias

    def __init__(self, data):
        self._data = data
        super().__init__()

    def set_alias(self, key, value):
        alias = Alias(self._data, value)
        super().__setitem__(key, alias)
        return alias

    def __setitem__(self, key, value):
        if key in self:
            alias = self._get_alias(key)
        else:
            alias = self.set_alias(key, key)
        alias.value = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            self.__setitem__(key, value)

    def __getitem__(self, item):
        alias = super().__getitem__(item)
        return alias.value

    def __getattr__(self, item):
        alias = super().__getattr__(item)
        return alias.value

    def _get_alias(self, item):
        alias = super().__getattr__(item)
        return alias

    def __str__(self):
        d = {k: v.value for k, v in self.items()}
        return d.__str__()

    def get_key(self, k):
        return self._get_alias(k)._key


class NetworkHandler(Handler):
    _type = torch.nn.Module
    _get_error_string = 'Model `{}` not found. You must add ' \
                        'it in `build_models` (as a dict entry).' \
                        ' Found: {}'

ResultsHandler = Handler


class LossHandler(Handler):
    '''Simple dict-like container for losses
    '''

    _type = torch.Tensor
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def __init__(self, nets):
        self._nets = nets
        super().__init__()

    def check_key_value(self, k, v):
        if k.startswith('_'):
            return True
        super().check_key_value(k, v)
        if k not in self._nets:
            raise AttributeError(
                'Keyword `{}` not in the model_handler. Found: {}.'.format(
                    k, tuple(
                        self._nets.keys())))

        if len(v.size()) > 0:
            raise ValueError('Loss must be a scalar. Got {}'.format(v.size()))

        return True
