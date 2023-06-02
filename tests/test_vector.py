import numpy as np
from dctkit.mesh import simplex, util
import dctkit.dec.vector as V


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

    v_coeffs = np.ones((S.S[2].shape[0], S.embedded_dim))
    v = V.DiscreteVectorFieldD(S, v_coeffs)
    c = V.flat(v)
    print(S.dedges)
    print(S.delements_areas)
    print(S.dedges_complete_areas)
    print(S.flat_coeffs_matrix)
    print(c.coeffs)
    assert False
