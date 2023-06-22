import gmsh
import numpy as np
from dctkit.mesh import util, simplex
from dctkit.math.opt import optctrl
from dctkit.apps.linear_elasticity import LinearElasticity
import dctkit as dt


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

    bnd_edges_idx = S.bnd_faces_indices
    left_bnd_nodes_idx, _ = util.get_nodes_from_physical_group(1, 2)
    right_bnd_nodes_idx, _ = util.get_nodes_from_physical_group(1, 3)
    left_bnd_edges_idx = util.get_belonging_elements(dim=1, tag=2, nodeTagsPerElem=S.S[1])
    right_bnd_edges_idx = util.get_belonging_elements(dim=1, tag=4, nodeTagsPerElem=S.S[1])

    # conversion to lists makes concatenation easier when assigning bcs
    left_bnd_nodes_idx = list(left_bnd_nodes_idx)
    right_bnd_nodes_idx = list(right_bnd_nodes_idx)
    bottom_left_corner = left_bnd_nodes_idx.pop(0)

    # Dirichlet bcs
    applied_strain = 0.02
    left_bnd_pos_components = [0]
    right_bnd_pos_components = [0]
    left_bnd_nodes_pos = S.node_coord[left_bnd_nodes_idx, :][:, left_bnd_pos_components]
    right_bnd_nodes_pos = S.node_coord[right_bnd_nodes_idx,
                                       :][:, right_bnd_pos_components]*(1.+applied_strain)
    bottom_left_corner_pos = S.node_coord[bottom_left_corner, :]

    boundary_values = {"0": (left_bnd_nodes_idx + right_bnd_nodes_idx,
                             np.vstack((left_bnd_nodes_pos, right_bnd_nodes_pos)).flatten()),
                       ":": (bottom_left_corner, bottom_left_corner_pos)}

    idx_free_edges = list(set(bnd_edges_idx) - set(right_bnd_edges_idx) - set(left_bnd_edges_idx))
    bnd_tractions_free_values = np.zeros((len(idx_free_edges), 2), dtype=dt.float_dtype)
    boundary_tractions = {':': (idx_free_edges, bnd_tractions_free_values)}

    mu_ = 1.
    lambda_ = 10.
    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    gamma = 1000.

    num_faces = S.S[2].shape[0]
    embedded_dim = S.embedded_dim

    f = np.zeros((num_faces, (embedded_dim-1)))
    f_flattened = f.flatten()
    node_coords_flattened = S.node_coord.flatten()

    obj_args = {'f': f_flattened,
                'gamma': gamma,
                'boundary_values': boundary_values,
                'boundary_tractions': boundary_tractions}

    prb = optctrl.OptimizationProblem(dim=S.node_coord.size,
                                      state_dim=S.node_coord.size,
                                      objfun=ela.obj_linear_elasticity)
    prb.set_obj_args(obj_args)
    curr_node_coords_flatten = prb.run(x0=node_coords_flattened)
    curr_node_coords = curr_node_coords_flatten.reshape(S.node_coord.shape)

    true_strain_xx = applied_strain
    true_strain_yy = -(lambda_/(2*mu_+lambda_))*applied_strain
    node_coords_final_true = S.node_coord.copy()
    node_coords_final_true[:, 0] *= 1+true_strain_xx
    node_coords_final_true[:, 1] *= 1+true_strain_yy

    error = np.sum((node_coords_final_true - curr_node_coords)**2)
    assert error < 1e-5
