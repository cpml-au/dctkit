from functools import partial
from jax import jit
import jax.ops as ops
from typing import Tuple
from jax import Array
import numpy.typing as npt


@partial(jit, static_argnums=(2, 3))
def spmm(A: Tuple[Array | npt.NDArray, Array | npt.NDArray, Array | npt.NDArray],
         v: Array | npt.NDArray, transpose=False, shape=None) -> Array:
    """Performs the matrix-matrix product between a sparse matrix in COO format and a
        dense matrix or column vector.

    Args:
        A: tuple (rows,cols,values) representing the sparse matrix in COO format.
        v: matrix or column vector.
        transpose: whether to transpose A before multiplication.
        shape: the number of rows of the matrix A.

    Returns:
        the result of the matrix-matrix product.
    """
    assert v.ndim > 1
    rows, cols, vals = A

    if transpose:
        vv = v.take(rows, axis=0)
    else:
        vv = v.take(cols, axis=0)

    # NOTE: make vals a column vector
    prod = vals[:, None] * vv

    if transpose:
        result = ops.segment_sum(prod, cols, shape)
    else:
        result = ops.segment_sum(prod, rows, shape)

    return result
