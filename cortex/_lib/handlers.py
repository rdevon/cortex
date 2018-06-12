'''Handlers.

'''

import logging

import torch

logger = logging.getLogger('cortex.handlers')


class Handler(dict):
    '''
    Simple dict-like container with support for `.` access
    Note: some of the functionalty might not work correctly
    as a dict, but so far simple tests pass.
    '''

    __delattr__ = dict.__delitem__
    _protected = dir({})
    _type = None
    _get_error_string = 'Keyword `{}` not found ' \
                        '(add as a dict entry). Found: {}'

    def check_key_value(self, k, v):
        if k.startswith('_'):
            return True
        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed.'
                             ' Only `{}` and subclasses are'
                             ' supported'.format(type(v), k, self._type))

        return True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().__init__(**kwargs)

    def __setitem__(self, k, v):
        passes = self.check_key_value(k, v)
        if passes:
            super().__setitem__(k, v)

    def __setattr__(self, k, v):
        if k.startswith('_'):
            super().__setattr__(k, v)
        else:
            self.__setitem__(k, v)

    def __getattr__(self, k):
        if k.startswith('_'):
            v = super().get(k)
            return v
        try:
            v = super().__getitem__(k)
        except KeyError:
            raise KeyError(
                self._get_error_string.format(
                    k, tuple(
                        self.keys())))
        return v

    def update(self, **kwargs):
        _kwargs = Handler()
        for k, v in kwargs.items():
            passes = self.check_key_value(k, v)
            if passes:
                _kwargs[k] = v
        super().update(**_kwargs)


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
        self._fn(value)
        super().__setitem__(key, value)


class Alias():
    '''An alias class for referencing objects in a base container.

    '''
    def __init__(self, data, key):
        self._data = data
        self._key = key

    @property
    def value(self):
        return self._data[self._key]

    @value.setter
    def value(self, value):
        self._data[self._key] = value


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
            alias = self.__getitem__(key)
        else:
            alias = self.set_alias(key, key)
        alias.value = value

    def __getitem__(self, item):
        alias = super().__getitem__(item)
        return alias.value

    def __str__(self):
        d = {k: v.value for k, v in self.items()}
        return d.__str__()


class NetworkHandler(Handler):
    '''
    Simple dict-like container for nn.Module's
    '''

    _type = torch.nn.Module
    _get_error_string = 'Model `{}` not found. You must add ' \
                        'it in `build_models` (as a dict entry).' \
                        ' Found: {}'

    def check_key_value(self, k, v):
        if k in self:
            logger.warning(
                'Key {} already in MODEL_HANDLER, ignoring.'.format(k))
            return False

        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if isinstance(v, (list, tuple)):
            for v_ in v:
                self.check_key_value(k, v_)
        elif self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. '
                             'Only `{}` and subclasses (or tuples '
                             'of {}) are supported'
                             .format(type(v), k, self._type, self._type))

        return True


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
