"""
Test suite for GAN network.
"""
import pytest
from cortex.built_ins.models.gan import *

def test_raise_measure_error():
    measure = 'X'
    with pytest.raises(NotImplementedError):
        raise_measure_error(measure)