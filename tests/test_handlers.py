'''Module for testing handlers.

'''

import torch

from cortex._lib.handlers import (AliasedHandler, PrefixedAliasedHandler,
                                  Handler, NetworkHandler)
from cortex.built_ins.networks.fully_connected import FullyConnectedNet


def test_basic_handler():
    h = Handler()
    h.a = 1
    assert h['a'] == 1

    h.a = 2

    h.lock()
    assert h._locked

    try:
        h.b = 10
        assert 0
    except KeyError:
        pass

    h = Handler(allow_overwrite=False)
    h.a = 1

    try:
        h.a = 2
        assert 0
    except KeyError:
        pass

    for kv in h.items():
        pass

    for k in h:
        pass


def test_network_handler():
    h = NetworkHandler()

    try:
        h.a = 1
        assert 0
    except TypeError:
        pass

    h.a = FullyConnectedNet(1, 2)

    assert isinstance(h.a, torch.nn.Module)

    h.b = FullyConnectedNet(2, 3)

    for k, m in h.items():
        assert isinstance(m, torch.nn.Module), k


def test_aliased_handler():
    h = Handler()

    aliases = dict(a='A', b='B', c='C', d='D')

    ah = AliasedHandler(h, aliases=aliases)

    ah.a = 13
    ah.b = 12

    try:
        ah.C = 22
        assert 0, 'Name in the set of aliases values cannot be set.'
    except KeyError:
        pass

    assert ah.a == 13
    assert h.A == 13
    assert ah.b == 12
    assert h.B == 12

    for k, v in ah.items():
        if k == 'a':
            pass
        elif k == 'b':
            pass
        else:
            assert False, k

    for k in ah:
        if k == 'a':
            pass
        elif k == 'b':
            pass
        else:
            assert False, k

    ah.pop('a')
    try:
        h.A
        assert 0
    except AttributeError:
        pass


def test_prefixed_handler():

    h = Handler()
    ah = PrefixedAliasedHandler(h, prefix='test')

    ah.a = 13
    ah.b = 12

    assert ah.a == 13
    assert h.test_a == 13
    assert ah.b == 12
    assert h.test_b == 12

    ah.pop('a')

    try:
        h.test_a
        assert 0
    except AttributeError:
        pass
