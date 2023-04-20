"""
    Dummy conftest.py for dctkit.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
import dctkit


@pytest.fixture()
def setup_test():
    # NOTE: running multiple JAX tests with different data types DOES NOT work
    # (jax_enable_x64 must be changed AT STARTUP)
    dctkit.config_called = False
    config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)
