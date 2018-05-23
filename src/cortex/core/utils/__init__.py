# -*- coding: utf-8 -*-
"""
:mod:`cortex.core.utils` -- Package-wide useful routines
=======================================================

.. module:: utils
   :platform: Unix
   :synopsis: Helper functions useful in possibly all :mod:`cortex.core`'s modules.
"""

from abc import ABCMeta
from glob import glob
from importlib import import_module
import logging
import os

import pkg_resources


log = logging.getLogger(__name__)


class SingletonType(type):
    """Metaclass that implements the singleton pattern for a Python class."""

    def __init__(cls, name, bases, dictionary):
        """Create a class instance variable and initiate it to None object."""
        super(SingletonType, cls).__init__(name, bases, dictionary)
        cls.instance = None

    def __call__(cls, *args, **kwargs):
        """Create an object if does not already exist, otherwise return what there is."""
        if cls.instance is None:
            cls.instance = super(SingletonType, cls).__call__(*args, **kwargs)
        elif args or kwargs:
            raise ValueError("A singleton instance has already been instantiated.")
        return cls.instance


class AbstractSingletonType(SingletonType, ABCMeta):
    """This will create singleton base classes, that need to be subclassed and implemented."""

    pass


class Factory(ABCMeta):
    """Instantiate appropriate wrapper for the infrastructure based on input
    argument, ``of_type``.

    Attributes
    ----------
    types : list of subclasses of ``cls.__base__``
       Updated to contain all possible implementations currently. Check out code.
    typenames : list of str
       Names of implemented wrapper classes, correspond to possible ``of_type``
       values.

    """

    def __init__(cls, names, bases, dictionary):
        """Search in directory for attribute names subclassing `bases[0]`"""
        super(Factory, cls).__init__(names, bases, dictionary)

        cls.modules = []
        base = import_module(cls.__base__.__module__)
        try:
            py_files = glob(os.path.abspath(os.path.join(base.__path__[0], '[A-Za-z]*.py')))
            py_mods = map(lambda x: '.' + os.path.split(os.path.splitext(x)[0])[1], py_files)
            for py_mod in py_mods:
                cls.modules.append(import_module(py_mod, package=cls.__base__.__module__))
        except AttributeError:
            # This means that base class and implementations reside in a module
            # itself and not a subpackage.
            pass

        # Get types advertised through entry points!
        for entry_point in pkg_resources.iter_entry_points(cls.__name__):
            entry_point.load()
            log.debug("Found a %s %s from distribution: %s=%s",
                      entry_point.name, cls.__name__,
                      entry_point.dist.project_name, entry_point.dist.version)

        # Get types visible from base module or package, but internal
        cls.types = cls.__base__.__subclasses__()
        cls.types = [class_ for class_ in cls.types if class_.__name__ != cls.__name__]
        cls.typenames = list(map(lambda x: x.__name__.lower(), cls.types))
        log.debug("Implementations found: %s", cls.typenames)

    def __call__(cls, of_type, *args, **kwargs):
        """Create an object, instance of ``cls.__base__``, on first call.

        :param of_type: Name of class, subclass of ``cls.__base__``, wrapper
           of a database framework that will be instantiated on the first call.
        :param args: positional arguments to initialize ``cls.__base__``'s instance (if any)
        :param kwargs: keyword arguments to initialize ``cls.__base__``'s instance (if any)

        .. seealso::
           `Factory.typenames` for values of argument `of_type`.

        .. seealso::
           Attributes of ``cls.__base__`` and ``cls.__base__.__init__`` for
           values of `args` and `kwargs`.

        .. note:: New object is saved as `Factory`'s internal state.

        :return: The object which was created on the first call.
        """
        for inherited_class in cls.types:
            if inherited_class.__name__.lower() == of_type.lower():
                return inherited_class.__call__(*args, **kwargs)

        error = "Could not find implementation of {0}, type = '{1}'".format(
            cls.__base__.__name__, of_type)
        error += "\nCurrently, there is an implementation for types:\n"
        error += str(cls.typenames)
        raise NotImplementedError(error)


class SingletonFactory(AbstractSingletonType, Factory):
    """Wrapping `Factory` with `SingletonType`. Keep compatibility with `AbstractSingletonType`."""

    pass

'''Utility methods

'''

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

import logging
import os

import numpy as np
import torch

logger = logging.getLogger('cortex.util')

try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1


def print_section(s):
    '''For printing sections to scripts nicely.
    Args:
        s (str): string of section
    '''
    h = s + ('-' * (_columns - len(s)))
    print(h)


class Handler(dict):
    '''
    Simple dict-like container with support for `.` access
    Note: some of the functionalty might not work correctly as a dict, but so far simple tests pass.
    '''

    __delattr__ = dict.__delitem__
    _protected = dir({})
    _type = None
    _get_error_string = 'Keyword `{}` not found (add as a dict entry). Found: {}'


    def check_key_value(self, k, v):
        if k in self._protected:
            raise KeyError('Keyword `{}` is protected.'.format(k))

        if self._type and not isinstance(v, self._type):
            raise ValueError('Type `{}` of `{}` not allowed. Only `{}` and subclasses are supported'.format(
                type(v), k, self._type))

        return True

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.check_key_value(k, v)
        super().__init__(**kwargs)

    def __setitem__(self, k, v):
        passes = self.check_key_value(k, v)
        if passes:
            super().__setitem__(k, v)

    def unsafe_set(self, k, v):
        super().__setitem__(k, v)

    def __setattr__(self, k, v):
        self.__setitem__(k, v)

    def __getattr__(self, k):
        if k.startswith('__'):
            return super.get(k)
        try:
            v = super().__getitem__(k)
        except KeyError:
            raise KeyError(self._get_error_string.format(k, tuple(self.keys())))
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


def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.items():
        if isinstance(v, dict):
            if k not in d_to_update.keys():
                d_to_update[k] = {}
            update_dict_of_lists(d_to_update[k], **v)
        elif k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]


def bad_values(d):
    failed = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v_ = bad_values(v)
            if v_:
                failed[k] = v_
        else:
            if isinstance(v, torch.Tensor):
                v_ = v.item()
            else:
                v_ = v
            if np.isnan(v_) or np.isinf(v_):
                failed[k] = v_

    if len(failed) == 0:
        return False
    return failed


def convert_to_numpy(o):
    if isinstance(o, torch.Tensor):
        o = o.data.cpu().numpy()
        if len(o.shape) == 1 and o.shape[0] == 1:
            o = o[0]
    elif isinstance(o, (torch.cuda.FloatTensor, torch.cuda.LongTensor)):
        o = o.cpu().numpy()
    elif isinstance(o, list):
        for i in range(len(o)):
            o[i] = convert_to_numpy(o[i])
    elif isinstance(o, tuple):
        o_ = tuple()
        for i in range(len(o)):
            o_ = o_ + (convert_to_numpy(o[i]),)
        o = o_
    elif isinstance(o, dict):
        for k in o.keys():
            o[k] = convert_to_numpy(o[k])
    return o


def compute_tsne(X, perplexity=40, n_iter=300, init='pca'):
    from sklearn.manifold import TSNE

    tsne = TSNE(2, perplexity=perplexity, n_iter=n_iter, init=init)
    points = X.tolist()
    return tsne.fit_transform(points)
