import dctkit
from dctkit.math import spmv
import numpy as np


def test_spmm(setup_test):
    int_dtype = dctkit.int_dtype
    float_dtype = dctkit.float_dtype
    rows = np.array([0, 0, 1, 2], dtype=int_dtype)
    cols = np.array([0, 1, 1, 2], dtype=int_dtype)
    vals = np.array([1., 2., 3., 5.], dtype=int_dtype)
    A = [rows, cols, vals]

    v = np.array([0., 1., 2.], dtype=float_dtype)
    v_matrix = (v*np.ones((5, 3), dtype=float_dtype)).T
    result_true = np.array([2., 3., 10.], dtype=float_dtype)
    result = spmv.spmm(A, v, shape=3)
    result_matrix = spmv.spmm(A, v_matrix, shape=3)
    result_matrix_true = (result*np.ones((5, 3), dtype=float_dtype)).T
    print(result)
    print(result_matrix)
    assert np.allclose(result, result_true)
    assert np.allclose(result_matrix, result_matrix_true)

    result_transpose_true = np.array([0., 3., 10.], dtype=float_dtype)
    result_transpose_matrix_true = (
        result_transpose_true*np.ones((5, 3), dtype=float_dtype)).T
    result = spmv.spmm(A, v, transpose=True, shape=3)
    result_matrix = spmv.spmm(A, v_matrix, transpose=True, shape=3)
    print(result)
    print(result_matrix)
    assert result.dtype == float_dtype
    assert np.allclose(result, result_transpose_true)
    assert np.allclose(result_matrix, result_transpose_matrix_true)
