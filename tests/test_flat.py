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
    S.get_flat_PDP_weights()
    S.get_flat_dual_upw_weights()
    S.get_S_dual()

    # test flat operators
    vP0_coeffs = np.ones((S.num_nodes, S.space_dim), dtype=dt.float_dtype)
    TP0_coeffs = np.ones((S.num_nodes, S.space_dim,
                          S.space_dim), dtype=dt.float_dtype)
    vD0_coeffs = np.ones((S.S[2].shape[0], S.space_dim), dtype=dt.float_dtype)
    TD0_coeffs = np.ones((S.S[2].shape[0], S.space_dim,
                          S.space_dim), dtype=dt.float_dtype)
    vP0 = C.CochainP0V(S, vP0_coeffs)
    TP0 = C.CochainP0V(S, TP0_coeffs)
    vD0 = C.CochainD0V(S, vD0_coeffs)
    TD0 = C.CochainD0T(S, TD0_coeffs)

    dedges = S.dual_edges_vectors[:, :vD0.coeffs.shape[1]]
    pedges = S.primal_edges_vectors[:, :vD0.coeffs.shape[1]]
    dual_edges_coch = C.CochainD1V(complex=S, coeffs=dedges)
    primal_edges_coch = C.CochainP1V(complex=S, coeffs=pedges)

    c_v_DPD = V.flat(vD0, S.flat_DPD_weights, dual_edges_coch)
    c_T_DPD = V.flat(TD0, S.flat_DPD_weights, dual_edges_coch)
    c_v_DPP = V.flat(vD0, S.flat_DPP_weights, primal_edges_coch)
    c_T_DPP = V.flat(TD0, S.flat_DPP_weights, primal_edges_coch)
    c_v_PDP = V.flat(vP0, S.flat_PDP_weights, primal_edges_coch)
    c_T_PDP = V.flat(TP0, S.flat_PDP_weights, primal_edges_coch)
    c_v_dual_upw = V.flat(vD0, S.flat_dual_upw_weights, dual_edges_coch)
    c_T_dual_upw = V.flat(TD0, S.flat_dual_upw_weights, dual_edges_coch)

    c_v_DPD_true_coeffs = S.dual_edges_vectors.sum(axis=1)[:, None]
    c_T_DPD_true_coeffs = np.ones((12, 3), dtype=dt.float_dtype)
    c_T_DPD_true_coeffs = c_v_DPD_true_coeffs*c_T_DPD_true_coeffs
    c_v_DPP_true_coeffs = S.primal_edges_vectors.sum(axis=1)[:, None]
    c_T_DPP_true_coeffs = np.ones((12, 3), dtype=dt.float_dtype)
    c_T_DPP_true_coeffs = c_v_DPP_true_coeffs*c_T_DPP_true_coeffs
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
    c_T_PDP_true_coeffs = np.array([[0.3660254,  0.3660254,  0.3660254],
                                    [-1.3660254, -1.3660254, -1.3660254],
                                    [-1., -1., -1.],
                                    [-1., -1., -1.],
                                    [-1.3660254, -1.3660254, -1.3660254],
                                    [-1.3660254, -1.3660254, -1.3660254],
                                    [-0.3660254, -0.3660254, -0.3660254],
                                    [-0.3660254, -0.3660254, -0.3660254],
                                    [1.,  1.,  1.],
                                    [1.,  1.,  1.],
                                    [1.3660254,  1.3660254,  1.3660254],
                                    [0.3660254,  0.3660254,  0.3660254]])
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
    c_T_dual_upw_true_coeffs = np.array([[0.,  0.,  0.],
                                         [0.,  0.,  0.],
                                         [-0.57735027, -0.57735027, -0.57735027],
                                         [0.,  0.,  0.],
                                         [0.21132487,  0.21132487,  0.21132487],
                                         [0.,  0.,  0.],
                                         [0.78867513,  0.78867513,  0.78867513],
                                         [0.,  0.,  0.],
                                         [0.57735027,  0.57735027,  0.57735027],
                                         [0.,  0.,  0.],
                                         [-0.21132487, -0.21132487, -0.21132487],
                                         [-0.78867513, -0.78867513, -0.78867513]])

    assert np.allclose(c_v_DPD.coeffs, c_v_DPD_true_coeffs)
    assert np.allclose(c_T_DPD.coeffs, c_T_DPD_true_coeffs)
    assert np.allclose(c_v_DPP.coeffs, c_v_DPP_true_coeffs)
    assert np.allclose(c_T_DPP.coeffs, c_T_DPP_true_coeffs)
    assert np.allclose(c_v_PDP.coeffs, c_v_PDP_true_coeffs)
    assert np.allclose(c_T_PDP.coeffs, c_T_PDP_true_coeffs)
    assert np.allclose(c_v_dual_upw.coeffs, c_v_dual_upw_true_coeffs)
    assert np.allclose(c_T_dual_upw.coeffs, c_T_dual_upw_true_coeffs)
