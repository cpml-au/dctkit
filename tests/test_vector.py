import numpy as np
import dctkit as dt
from dctkit.mesh import simplex, util
import dctkit.dec.vector as V


def test_vector_cochain(setup_test):
    _, _, S_2, node_coords, bnd_faces_tags = util.generate_hexagon_mesh(1, 1)
    S = simplex.SimplicialComplex(S_2, node_coords, bnd_faces_tags=bnd_faces_tags)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    S.get_dual_edge_vectors()
    S.get_flat_weights()

    # test flat operator
    v_coeffs = np.ones((S.embedded_dim, S.S[2].shape[0]), dtype=dt.float_dtype)
    T_coeffs = np.ones((S.embedded_dim, S.embedded_dim,
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
