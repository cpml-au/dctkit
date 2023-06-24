import pytest
from dctkit import config, FloatDtype, Platform
import dctkit


@pytest.fixture()
def setup_test():
    # NOTE: running multiple JAX tests with different data types DOES NOT work
    # (jax_enable_x64 must be changed AT STARTUP)
    # WARNING: test_elastica fails with float32 precision
    dctkit.config_called = False
    config(FloatDtype.float64, Platform.cpu)
