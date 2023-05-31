import sys
import enum
import jax.numpy as jnp
from jax.config import config as cfg

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

FloatDtype = enum.Enum('FloatDtype', ['float32', 'float64'])
IntDtype = enum.Enum('IntDtype', ['int32', 'int64'])
Platform = enum.Enum('Platform', ['cpu', 'gpu'])

# data types, backend and platform used in all the modules
float_dtype = FloatDtype.float64.name
int_dtype = IntDtype.int64.name
backend = jnp
platform = Platform.cpu
config_called = False


def config(fdtype=FloatDtype.float64, platfm=Platform.cpu):
    """Set global configuration parameters."""
    global config_called, float_dtype, int_dtype, platform
    if not config_called:
        float_dtype = fdtype.name
        platform = platfm

        cfg.update('jax_platform_name', platfm.name)

        if fdtype == FloatDtype.float64:
            int_dtype = IntDtype.int64.name
            cfg.update("jax_enable_x64", True)
        else:
            int_dtype = IntDtype.int32.name

        config_called = True
