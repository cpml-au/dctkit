from functools import partial
from jax import jit
import jax.ops as ops
from typing import Tuple
from jax import Array
import numpy.typing as npt
from jax.experimental import sparse


@partial(jit, static_argnums=(2, 3))
def spmv_coo(A: Tuple[Array | npt.NDArray, Array | npt.NDArray, Array | npt.NDArray],
             v: Array | npt.NDArray, transpose=False, shape=None) -> Array:
    """Performs the matrix-vector product between a sparse matrix in COO format and a
        vector.

    Args:
        A: tuple (rows,cols,values) representing the sparse matrix in COO format.
        v: vector.
        transpose: whether to transpose A before multiplication.
        shape

    Returns:
        the vector resulting from the matrix-vector product.
    """
    rows, cols, vals = A

    if transpose:
        vv = v.take(rows)
    else:
        vv = v.take(cols)

    prod = vals * vv

    if transpose:
        result = ops.segment_sum(prod, cols, shape)
    else:
        result = ops.segment_sum(prod, rows, shape)

    return result


@partial(jit, static_argnums=(2,))
def spmv_bcoo_jax(A: sparse.BCOO, v: Array | npt.NDArray, transpose=False) -> Array:

    if transpose:
        return A.T @ v
    else:
        return A @ v
