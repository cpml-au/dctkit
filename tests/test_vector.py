import numpy as np
import dctkit as dt
from dctkit.mesh import util
import dctkit.dec.flat as V
from dctkit.dec import cochain as C


def test_flat(setup_test):
    mesh, _ = util.generate_hexagon_mesh(1., 1.)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPP_weights()
    S.get_flat_DPD_weights()

    # test flat operators
    v_coeffs = np.ones((S.space_dim, S.S[2].shape[0]), dtype=dt.float_dtype)
    T_coeffs = np.ones((S.space_dim, S.space_dim,
                       S.S[2].shape[0]), dtype=dt.float_dtype)
    v = C.CochainD0V(S, v_coeffs)
    T = C.CochainD0T(S, T_coeffs)

    c_v_DPD = V.flat_DPD(v)
    c_T_DPD = V.flat_DPD(T)
    c_v_DPP = V.flat_DPP(v)
    c_T_DPP = V.flat_DPP(T)

    c_v_DPD_true_coeffs = S.dual_edges_vectors.sum(axis=1)
    c_T_DPD_true_coeffs = np.ones((12, 3), dtype=dt.float_dtype)
    c_T_DPD_true_coeffs = c_v_DPD_true_coeffs[:, None]*c_T_DPD_true_coeffs
    c_v_DPP_true_coeffs = S.primal_edges_vectors.sum(axis=1)
    c_T_DPP_true_coeffs = np.ones((12, 3), dtype=dt.float_dtype)
    c_T_DPP_true_coeffs = c_v_DPP_true_coeffs[:, None]*c_T_DPP_true_coeffs

    assert np.allclose(c_v_DPD.coeffs, c_v_DPD_true_coeffs)
    assert np.allclose(c_T_DPD.coeffs, c_T_DPD_true_coeffs)
    assert np.allclose(c_v_DPP.coeffs, c_v_DPP_true_coeffs)
    assert np.allclose(c_T_DPP.coeffs, c_T_DPP_true_coeffs)
