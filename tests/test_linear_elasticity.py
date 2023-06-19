import gmsh
import numpy as np
from dctkit.mesh import util, simplex
from dctkit.math.opt import optctrl
from dctkit.apps.linear_elasticity import LinearElasticity
import dctkit as dt
import jax.numpy as jnp


def test_linear_elasticity(setup_test):
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
    gamma = 1000.
    num_faces = S.S[2].shape[0]
    embedded_dim = S.embedded_dim

    # set displacement boundary conditions
    # left_bnd_nodes_idx = [1, 3, 5]
    # right_bnd_nodes_idx = [0, 2, 7]
    # left_bnd_nodes_pos = S.node_coord[left_bnd_nodes_idx, :]
    # right_bnd_nodes_pos = S.node_coord[right_bnd_nodes_idx, :]
    # on the left edge, all the positions are blocked; on the right
    # edge x-coordinate is multiplied by 1.1 while y-coordinate is fixed
    # right_bnd_nodes_pos[:, 0] *= 1.1
    # clamped_bnd_pos = np.vstack((left_bnd_nodes_pos, right_bnd_nodes_pos))
    # clamped_bnd_idx = left_bnd_nodes_idx + right_bnd_nodes_idx
    # boundary_values = (clamped_bnd_idx, clamped_bnd_pos)
    idx = [1, 3, 5]
    value = S.node_coord[idx, :]
    boundary_values = (idx, value)

    # set tractions boundary conditions
    idx_tractions = jnp.array([0, 3, 6, 10])
    bnd_tractions_values = jnp.zeros((4, 2), dtype=dt.float_dtype)
    boundary_tractions = (idx_tractions, bnd_tractions_values)

    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    f = np.zeros((num_faces, (embedded_dim-1)))
    f[[3, 6], 0] = 0.6*np.ones(2)
    f = f.flatten()
    node_coords_reshaped = S.node_coord.flatten()

    obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values,
                'boundary_tractions': boundary_tractions}

    prb = optctrl.OptimizationProblem(dim=S.node_coord.size,
                                      state_dim=S.node_coord.size,
                                      objfun=ela.obj_linear_elasticity)
    prb.set_obj_args(obj_args)
    curr_node_coords = prb.run(x0=node_coords_reshaped)
    print(prb.last_opt_result)
    curr_node_coords_reshape = curr_node_coords.reshape(S.node_coord.shape)
    node_coords_final_true = S.node_coord.copy()
    node_coords_final_true[:, 0] *= 1.1
    epsilon = ela.get_strain(curr_node_coords_reshape)
    stress = ela.get_stress(epsilon)
    print("-------------------------------------")
    print(ela.obj_linear_elasticity(node_coords_final_true,
          f, gamma, boundary_values, boundary_tractions))
    print(curr_node_coords_reshape)
    print("-------------------------------------")
    print("eps=", epsilon)
    print("stress=", stress)
    error = np.sum((node_coords_final_true - curr_node_coords_reshape)**2)
    assert error < 1e-1
    assert False
