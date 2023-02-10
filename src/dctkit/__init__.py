import sys
import enum

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
# FIXME: we always use jax as a backend
Backend = enum.Enum('Backend', ['numpy', 'jax'])
Platform = enum.Enum('Platform', ['cpu', 'gpu'])

# glabal data type for the library
float_dtype = FloatDtype.float32.name
int_dtype = IntDtype.int32.name
backend = Backend.jax
platform = Platform.cpu


def config(fdtype=FloatDtype.float32, idtype=IntDtype.int32, backnd=Backend.jax, platfm=Platform.cpu):
    """Set global configuration parameters."""
    global float_dtype, int_dtype, backend, platform
    float_dtype = fdtype.name
    int_dtype = idtype.name
    backend = backnd
    platform = platfm
    if backnd == Backend.jax:
        from jax.config import config
        config.update('jax_platform_name', platfm.name)

        if fdtype == FloatDtype.float64:
            config.update("jax_enable_x64", True)
