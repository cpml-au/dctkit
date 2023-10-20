import numpy as np
import dctkit as dt
from dctkit.mesh import util
import dctkit.dec.vector as V
from dctkit.dec import cochain as C


def test_vector(setup_test):
    mesh, _ = util.generate_hexagon_mesh(1., 1.)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPP_weights()
    S.get_flat_DPD_weights()

    # test flat operators
    v_coeffs = np.ones((S.space_dim, S.S[2].shape[0]), dtype=dt.float_dtype)
    T_coeffs = np.ones((S.space_dim, S.space_dim,
                       S.S[2].shape[0]), dtype=dt.float_dtype)
    v = V.DiscreteVectorFieldD(S, v_coeffs)
    T = V.DiscreteTensorFieldD(S, T_coeffs, 2)

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


def test_flat_PDD(setup_test):
    num_x_points = 11
    x_max = 1
    mesh, _ = util.generate_line_mesh(num_x_points, x_max)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()

    c = C.CochainD0(S, np.arange(10))
    flat_PDD_explicit = V.flat_PDD(c, "upwind")
    flat_PDD_implicit = V.flat_PDD_2(c)

    assert np.allclose(flat_PDD_explicit.coeffs, flat_PDD_implicit.coeffs)
