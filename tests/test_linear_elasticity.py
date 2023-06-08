import gmsh
import numpy as np
from dctkit.mesh import util, simplex
from dctkit.math.opt import optctrl
from dctkit.apps.linear_elasticity import LinearElasticity
import dctkit as dt
from jax import grad


def test_linear_elasticity():
    lc = 0.5
    _, _, S_2, node_coords, bnd_faces_tags = util.generate_square_mesh(lc)
    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    bnodes = bnodes.astype(dt.int_dtype)
    S = simplex.SimplicialComplex(
        S_2, node_coords, bnd_faces_tags=bnd_faces_tags, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    S.get_dual_edge_vectors()
    S.get_flat_weights()

    mu_ = 1.
    lambda_ = 10.
    gamma = 10000.
    num_faces = S.S[2].shape[0]
    embedded_dim = S.embedded_dim
    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    f = np.zeros(num_faces*(embedded_dim-1))
    node_coords_reshaped = S.node_coord.flatten()
    idx = bnodes[:2]
    value = S.node_coord[idx, :]
    boundary_values = (idx, value)
    obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values}

    prb = optctrl.OptimizationProblem(dim=S.node_coord.size,
                                      state_dim=S.node_coord.size,
                                      objfun=ela.obj_linear_elasticity)
    print(ela.obj_linear_elasticity(node_coords_reshaped, f, gamma, boundary_values))
    print(grad(ela.obj_linear_elasticity)(
        node_coords_reshaped, f, gamma, boundary_values))
    prb.set_obj_args(obj_args)
    sol = prb.run(x0=node_coords_reshaped)
    print(prb.last_opt_result)
    print(sol.reshape(S.node_coord.shape))
    print(S.node_coord)
    assert False
