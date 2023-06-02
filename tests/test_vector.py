import numpy as np
from dctkit.mesh import simplex, util
import dctkit.dec.vector as V
import jax.numpy as jnp


def test_vector(setup_test):
    _, _, S_2, node_coords, belem_tags = util.generate_hexagon_mesh(1, 1)
    S = simplex.SimplicialComplex(S_2, node_coords, belem_tags=belem_tags)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    S.get_dual_edges()
    S.get_areas_complementary_duals()
    S.get_flat_coeffs_matrix()

    # test flat operator
    v_coeffs = np.ones((S.S[2].shape[0], S.embedded_dim))
    v = V.DiscreteVectorFieldD(S, v_coeffs)
    c = V.flat(v)
    c_true_coeffs = S.dedges.sum(axis=1)

    assert jnp.allclose(c.coeffs, c_true_coeffs)
