from dctkit.math import spmv
import numpy as np


def test_spmv_coo():
    rows = np.array([0, 0, 1, 2], dtype=np.int32)
    cols = np.array([0, 1, 1, 2], dtype=np.int32)
    vals = np.array([1., 2., 3., 5.], dtype=np.float32)
    A = [rows, cols, vals]

    v = np.array([0., 1., 2.], dtype=np.float32)
    result_true = np.array([2., 3., 10.], dtype=np.float32)
    result = spmv.spmv_coo(A, v, shape=3)
    print(result)
    assert np.allclose(result, result_true)

    result_transpose_true = np.array([0., 3., 10.], dtype=np.float32)
    result = spmv.spmv_coo(A, v, transpose=True, shape=3)
    print(result)
    print(result.dtype)
    assert np.allclose(result, result_transpose_true)


if __name__ == '__main__':
    test_spmv_coo()
