'''Module for testing handlers.

'''

import torch

from cortex._lib.handlers import Handler, NetworkHandler
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


def test_network_handler():
    h = NetworkHandler()

    try:
        h.a = 1
        assert 0
    except:
        pass

    h.a = FullyConnectedNet(1, 2)

    assert isinstance(h.a, torch.nn.Module)

    h.b = FullyConnectedNet(2, 3)

    for k, m in h.items():
        assert isinstance(m, torch.nn.Module), k

