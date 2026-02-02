import numpy as np
import dctkit as dt
from dctkit.mesh import util
from dctkit.dec.flat import flat_DPD, flat_DPP, flat_PDP, flat_dual_upw
from dctkit.dec import cochain as C


def test_flat_2D(setup_test):
    mesh, _ = util.generate_hexagon_mesh(1., 1.)
    S = util.build_complex_from_mesh(mesh, space_dim=2)
    S.get_hodge_star()
    S.get_flat_DPP_weights()
    S.get_flat_DPD_weights()
    S.get_flat_PDP_weights()
    S.get_flat_dual_upw_weights()

    # vector-valued cochains test
    vP0_coeffs = np.ones((S.num_nodes, S.space_dim), dtype=dt.float_dtype)
    vD0_coeffs = np.ones((S.S[2].shape[0], S.space_dim), dtype=dt.float_dtype)
    vP0 = C.CochainP0V(S, vP0_coeffs)
    vD0 = C.CochainD0V(S, vD0_coeffs)

    c_v_DPD = flat_DPD(vD0)
    c_v_DPP = flat_DPP(vD0)
    c_v_PDP = flat_PDP(vP0)
    c_v_dual_upw = flat_dual_upw(vD0)

    c_v_DPD_true_coeffs = S.dual_edges_vectors.sum(axis=1)[:, None]
    c_v_DPP_true_coeffs = S.primal_edges_vectors.sum(axis=1)[:, None]
    c_v_PDP_true_coeffs = np.array([[0.3660254],
                                    [-1.3660254],
                                    [-1.],
                                    [-1.],
                                    [-1.3660254],
                                    [-1.3660254],
                                    [-0.3660254],
                                    [-0.3660254],
                                    [1.],
                                    [1.],
                                    [1.3660254],
                                    [0.3660254]])
    c_v_dual_upw_true_coeffs = np.array([[0.],
                                         [0.],
                                         [-0.57735027],
                                         [0.],
                                         [0.21132487],
                                         [0.],
                                         [0.78867513],
                                         [0.],
                                         [0.57735027],
                                         [0.],
                                         [-0.21132487],
                                         [-0.78867513]])

    assert np.allclose(c_v_DPD.coeffs, c_v_DPD_true_coeffs)
    assert np.allclose(c_v_DPP.coeffs, c_v_DPP_true_coeffs)
    assert np.allclose(c_v_PDP.coeffs, c_v_PDP_true_coeffs)
    assert np.allclose(c_v_dual_upw.coeffs, c_v_dual_upw_true_coeffs)

    # tensor-valued cochains test
    TP0_coeffs = np.ones((S.num_nodes, S.space_dim,
                          S.space_dim), dtype=dt.float_dtype)
    TD0_coeffs = np.ones((S.S[2].shape[0], S.space_dim,
                          S.space_dim), dtype=dt.float_dtype)
    TP0 = C.CochainP0V(S, TP0_coeffs)
    TD0 = C.CochainD0T(S, TD0_coeffs)

    c_T_DPD = flat_DPD(TD0)
    c_T_DPP = flat_DPP(TD0)
    c_T_PDP = flat_PDP(TP0)
    c_T_dual_upw = flat_dual_upw(TD0)

    c_T_DPD_true_coeffs = np.ones((12, 2), dtype=dt.float_dtype)
    c_T_DPD_true_coeffs = c_v_DPD_true_coeffs*c_T_DPD_true_coeffs
    c_T_DPP_true_coeffs = np.ones((12, 2), dtype=dt.float_dtype)
    c_T_DPP_true_coeffs = c_v_DPP_true_coeffs*c_T_DPP_true_coeffs
    c_T_PDP_true_coeffs = np.array([[0.3660254,  0.3660254],
                                    [-1.3660254, -1.3660254],
                                    [-1., -1.],
                                    [-1., -1.],
                                    [-1.3660254, -1.3660254],
                                    [-1.3660254, -1.3660254],
                                    [-0.3660254, -0.3660254],
                                    [-0.3660254, -0.3660254],
                                    [1.,  1.],
                                    [1.,  1.],
                                    [1.3660254,  1.3660254],
                                    [0.3660254,  0.3660254]])
    c_T_dual_upw_true_coeffs = np.array([[0.,  0.],
                                         [0.,  0.],
                                         [-0.57735027, -0.57735027],
                                         [0.,  0.],
                                         [0.21132487,  0.21132487],
                                         [0.,  0.],
                                         [0.78867513,  0.78867513],
                                         [0.,  0.],
                                         [0.57735027,  0.57735027],
                                         [0.,  0.],
                                         [-0.21132487, -0.21132487],
                                         [-0.78867513, -0.78867513]])

    assert np.allclose(c_T_DPD.coeffs, c_T_DPD_true_coeffs)
    assert np.allclose(c_T_DPP.coeffs, c_T_DPP_true_coeffs)
    assert np.allclose(c_T_PDP.coeffs, c_T_PDP_true_coeffs)
    assert np.allclose(c_T_dual_upw.coeffs, c_T_dual_upw_true_coeffs)
