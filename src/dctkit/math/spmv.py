from functools import partial
from jax import jit
import jax.ops as ops
from jax.config import config
# config.update('jax_platform_name', 'cpu')
config.update("jax_enable_x64", True)


# @profile
@partial(jit, static_argnums=(2, 3))
def spmv_coo(A, v, transpose=False, shape=None):
    """Performs the matrix-vector product between a sparse matrix in COO
    format and a vector.

    Args:
        A: tuple (rows,cols,values) representing the sparse matrix in
            COO format. Each of the elements of the tuple is a numpy.array or
            jax.array.
        v: vector (numpy.array or jax.array)

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
