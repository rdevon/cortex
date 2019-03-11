'''Handlers.

'''

from collections.abc import MutableMapping
import logging

import torch

from cortex._lib import exp


logger = logging.getLogger('cortex.handlers')


class Handler(MutableMapping):
    _type = None
    _get_error_string = 'Attribute `{}` not found. Available: {}'

    def __init__(self, allow_overwrite=True, **kwargs):
        self._allow_overwrite = allow_overwrite
        self._locked = False
        self._storage = dict(**kwargs)

    def _check_keyvalue(self, key, value):
        if self._type and not isinstance(value, self._type):
            raise TypeError('Invalid type ({}), expected {}.'
                            .format(type(value), self._type))

    def __getitem__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            d = dict((k, v) for k, v in self.__dict__.items()
                     if not k.startswith('_'))
            raise KeyError(self._get_error_string.format(key, tuple(d.keys())))

    def __delitem__(self, key):
        del self.__dict__[key]

    def __setitem__(self, key, value):
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_') or callable(value):
            return super().__setattr__(key, value)

        self._check_keyvalue(key, value)

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

    def __str__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return d.__str__()

    def __repr__(self):
        d = dict((k, v) for k, v in self.__dict__.items()
                 if not k.startswith('_'))
        return d.__repr__()


def convert_nested_dict_to_handler(d, _class=Handler):
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = convert_nested_dict_to_handler(v)

    return _class(**d)


class NestedNetworkHandler(Handler):
    def __init__(self, handler, model):
        self._model = model
        self._handler = handler
        self._aliases = {}

    def load(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __getattr__(self, item):
        if item.startswith('_'):
            return super().__getitem__(item)
        item = self._aliases.get(item, item)
        return self._handler.__getitem__(item)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        alias = None
        for k, v in self._handler.items():
            if value is v:
                alias = k
                break

        if alias:
            self._aliases[key] = alias
            return self._handler[alias]
        else:
            if isinstance(value, torch.nn.DataParallel):
                value = value.to(exp.DEVICE)
            else:
                value = torch.nn.DataParallel(value, device_ids=exp.DEVICE_IDS).to(exp.DEVICE)
            alias = '{}.{}'.format(self._model.name, key)
            self._aliases[key] = alias
            return self._handler.__setattr__(alias, value)

    def __getitem__(self, item):
        if item.startswith('_'):
            return super().__getitem__(item)
        item = self._aliases.get(item, item)
        return self._handler.__getitem__(item)

    def __setitem__(self, key, value):
        if key.startswith('_'):
            return super().__setitem__(key, value)

        alias = None
        for k, v in self._handler.items():
            if value is v:
                alias = k
                break

        if alias:
            self._aliases[key] = alias
            return self._handler[alias]
        else:
            if isinstance(value, torch.nn.DataParallel):
                value = value.to(exp.DEVICE)
            else:
                value = torch.nn.DataParallel(value, device_ids=exp.DEVICE_IDS).to(exp.DEVICE)
            alias = '{}.{}'.format(self._model.name, key)
            self._aliases[key] = alias
            return self._handler.__setitem__(alias, value)


def nested(handler, model):
    return NestedNetworkHandler(handler, model)


class NetworkHandler(Handler):
    _type = torch.nn.Module
    _get_error_string = 'Model `{}` not found. You must add ' \
                        'it in `build_models` (as a dict entry).' \
                        ' Found: {}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded = dict()
        self._lax_reload = False

    def load(self, lax_reload, **kwargs):
        self._lax_reload = lax_reload
        self._loaded.update(**kwargs)

    def __setitem__(self, key, value):
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if key in self._loaded:
            logger.debug('Loading parameters from saved model for {}'.format(key))
            MutableMapping.__setattr__(self, key, value)
            loaded = self._loaded[key]
            self.__dict__[key].load_state_dict(loaded.state_dict(), strict=not(self._lax_reload))
        elif not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        else:
            self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return MutableMapping.__setattr__(self, key, value)

        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if key in self._loaded:
            logger.debug('Loading parameters from saved model for {}'.format(key))
            MutableMapping.__setattr__(self, key, value)
            loaded = self._loaded[key]
            self.__dict__[key].load_state_dict(loaded.state_dict(), strict=not(self._lax_reload))
        elif not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        else:
            MutableMapping.__setattr__(self, key, value)


class ResultsHandler(Handler):

    def __setitem__(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if key.startswith('_'):
            return super().__setattr__(key, value)

        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')

        if not self._allow_overwrite and hasattr(self, key):
            raise KeyError('Overwriting keys not allowed.')

        super().__setattr__(key, value)


class LossHandler(Handler):
    '''Simple dict-like container for losses
    '''

    _type = (torch.Tensor, list)
    _get_error_string = 'Loss `{}` not found. You must add it as a dict entry'

    def __init__(self, nets, *args, method='append', **kwargs):
        self._nets = nets
        if method not in ('append', 'overwrite', 'add'):
            raise ValueError(method)
        self._method = method
        super().__init__(*args, **kwargs)

    def _check_keyvalue(self, k, v):
        if isinstance(v, (list, tuple)):
            for v_ in v:
                super()._check_keyvalue(k, v_)
            if len(v_.size()) > 0:
                raise ValueError(
                    'Loss must be a scalar. Got {}'.format(v_.size()))
        else:
            super()._check_keyvalue(k, v)
            if len(v.size()) > 0:
                raise ValueError(
                    'Loss must be a scalar. Got {}'.format(v.size()))

        if k not in self._nets:
            raise AttributeError(
                'Keyword `{}` not in the model_handler. Found: {}.'.format(
                    k, tuple(self._nets.keys())))

        return True

    def __setitem__(self, key, value):
        self._check_keyvalue(key, value)

        if self._locked:
            raise KeyError('Handler is locked.')
        self.__dict__[key] = value

    def __setattr__(self, key, value):
        if key.startswith('_'):
            return super().__setattr__(key, value)

        self._check_keyvalue(key, value)
        super().__setattr__(key, value)
