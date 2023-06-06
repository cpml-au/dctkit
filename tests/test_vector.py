import numpy as np
from dctkit.mesh import simplex, util
import dctkit.dec.vector as V
import jax.numpy as jnp


def test_vector(setup_test):
    _, _, S_2, node_coords, bnd_faces_tags = util.generate_hexagon_mesh(1, 1)
    S = simplex.SimplicialComplex(S_2, node_coords, bnd_faces_tags=bnd_faces_tags)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    S.get_dual_edge_vectors()
    S.get_flat_weights()

    # test flat operator
    v_coeffs = np.ones((S.embedded_dim, S.S[2].shape[0]))
    v = V.DiscreteVectorFieldD(S, v_coeffs)
    c = V.flat_DPD(v)
    c_true_coeffs = S.dual_edges_vectors.sum(axis=1)

    assert jnp.allclose(c.coeffs, c_true_coeffs)
