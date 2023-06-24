import numpy as np
import dctkit as dt
from dctkit.mesh import util
import dctkit.dec.vector as V


def test_vector_cochain(setup_test):
    mesh, _ = util.generate_hexagon_mesh(1., 1.)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_dual_edge_vectors()
    S.get_flat_weights()

    # test flat operator
    v_coeffs = np.ones((S.space_dim, S.S[2].shape[0]), dtype=dt.float_dtype)
    T_coeffs = np.ones((S.space_dim, S.space_dim,
                       S.S[2].shape[0]), dtype=dt.float_dtype)
    v = V.DiscreteVectorFieldD(S, v_coeffs)
    T = V.DiscreteTensorFieldD(S, T_coeffs, 2)
    c_v = V.flat_DPD(v)
    c_T = V.flat_DPD(T)
    c_v_true_coeffs = S.dual_edges_vectors.sum(axis=1)
    c_T_true_coeffs = np.ones((12, 3), dtype=dt.float_dtype)
    c_T_true_coeffs = c_v_true_coeffs[:, None]*c_T_true_coeffs

    assert np.allclose(c_v.coeffs, c_v_true_coeffs)
    assert np.allclose(c_T.coeffs, c_T_true_coeffs)
