import dctkit
from dctkit.math import spmv
import numpy as np


def test_spmv_coo(setup_test):
    int_dtype = dctkit.int_dtype
    float_dtype = dctkit.float_dtype
    rows = np.array([0, 0, 1, 2], dtype=int_dtype)
    cols = np.array([0, 1, 1, 2], dtype=int_dtype)
    vals = np.array([1., 2., 3., 5.], dtype=int_dtype)
    A = [rows, cols, vals]

    v = np.array([0., 1., 2.], dtype=float_dtype)
    result_true = np.array([2., 3., 10.], dtype=float_dtype)
    result = spmv.spmv_coo(A, v, shape=3)
    print(result)
    assert np.allclose(result, result_true)

    result_transpose_true = np.array([0., 3., 10.], dtype=float_dtype)
    result = spmv.spmv_coo(A, v, transpose=True, shape=3)
    print(result)
    assert result.dtype == float_dtype
    assert np.allclose(result, result_transpose_true)
