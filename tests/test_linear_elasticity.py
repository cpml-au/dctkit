import gmsh
import numpy as np
from dctkit.mesh import util, simplex
from dctkit.math.opt import optctrl
from dctkit.apps.linear_elasticity import LinearElasticity
import dctkit.dec.cochain as C
import dctkit as dt
from jax import grad


def test_linear_elasticity(setup_test):
    lc = 0.5
    _, _, S_2, node_coords, bnd_faces_tags = util.generate_square_mesh(lc)
    # _, _, S_2, node_coords, bnd_faces_tags = util.generate_hexagon_mesh(1, 1)
    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    print(bnodes)
    print(S_2)
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
    # FIXME: temporary setting
    idx = [1, 3, 5]
    value = S.node_coord[idx, :]
    num_faces = S.S[2].shape[0]
    embedded_dim = S.embedded_dim
    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    f = np.zeros((num_faces, (embedded_dim-1)))
    # FIXME: temporary setting
    f[[3, 6], 0] = 1.1*np.ones(2)
    # f[:, 1] = 0.1*np.ones(num_faces)
    f = f.flatten()
    node_coords_reshaped = S.node_coord.flatten()
    # idx = bnodes[-2:]
    # value = S.node_coord[idx, :]
    boundary_values = (idx, value)
    obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values}

    prb = optctrl.OptimizationProblem(dim=S.node_coord.size,
                                      state_dim=S.node_coord.size,
                                      objfun=ela.obj_linear_elasticity)
    prb.set_obj_args(obj_args)
    sol = prb.run(x0=node_coords_reshaped, maxeval=5000)
    sol_reshape = sol.reshape(S.node_coord.shape)
    print(prb.last_opt_result)
    print(sol.reshape(S.node_coord.shape))
    node_coords_final = S.node_coord.copy()
    node_coords_final[:, 0] *= 1.1
    node_coords_final_flat = node_coords_final.flatten()
    epsilon = ela.get_strain(node_coords_final)
    stress = ela.get_stress(epsilon)
    print("-------------------------------------")
    print(S.S[2])
    print(S.node_coord)
    print(S.S[1])
    print("-------------------------------------")
    print("eps=", epsilon)
    print("stress=", stress)
    np.save("ref.npy", S.node_coord)
    np.save("ela.npy", sol_reshape)
    np.save("conn.npy", S.S[2])
    assert False
