import numpy as np
from dctkit.mesh import util
from dctkit.math.opt import optctrl
from dctkit.physics.elasticity import LinearElasticity
import dctkit as dt
import pygmsh


def test_linear_elasticity_primal(setup_test):
    lc = 0.5
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPD_weights()

    ref_node_coords = S.node_coords

    bnd_edges_idx = S.bnd_faces_indices
    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")
    left_bnd_edges_idx = util.get_edges_for_physical_group(S, mesh, "left")
    right_bnd_edges_idx = util.get_edges_for_physical_group(S, mesh, "right")

    bottom_left_corner = left_bnd_nodes_idx.pop(0)

    mu_ = 1.
    lambda_ = 10.
    true_strain_xx = 0.02
    true_strain_yy = -(lambda_/(2*mu_+lambda_))*true_strain_xx
    true_curr_node_coords = S.node_coords.copy()
    true_curr_node_coords[:, 0] *= 1 + true_strain_xx
    true_curr_node_coords[:, 1] *= 1 + true_strain_yy
    left_bnd_pos_components = [0]
    right_bnd_pos_components = [0]

    left_bnd_nodes_pos = ref_node_coords[left_bnd_nodes_idx,
                                         :][:, left_bnd_pos_components]
    right_bnd_nodes_pos = true_curr_node_coords[right_bnd_nodes_idx,
                                                :][:, right_bnd_pos_components]
    bottom_left_corner_pos = ref_node_coords[bottom_left_corner, :]

    # NOTE: without flatten it does not work properly when concatenating multiple bcs;
    # fix this so that flatten is not needed (not intuitive)
    boundary_values = {"0": (left_bnd_nodes_idx + right_bnd_nodes_idx,
                             np.vstack((left_bnd_nodes_pos,
                                        right_bnd_nodes_pos)).flatten()),
                       ":": (bottom_left_corner, bottom_left_corner_pos)}

    idx_free_edges = list(set(bnd_edges_idx) -
                          set(right_bnd_edges_idx) - set(left_bnd_edges_idx))
    bnd_tractions_free_values = np.zeros((len(idx_free_edges), 2), dtype=dt.float_dtype)
    boundary_tractions = {':': (idx_free_edges, bnd_tractions_free_values)}

    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    gamma = 1000.

    num_faces = S.S[2].shape[0]
    embedded_dim = S.space_dim
    f = np.zeros((num_faces, (embedded_dim-1))).flatten()

    obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values,
                'boundary_tractions': boundary_tractions}

    prb = optctrl.OptimizationProblem(dim=S.node_coords.size,
                                      state_dim=S.node_coords.size,
                                      objfun=ela.obj_linear_elasticity)

    prb.set_obj_args(obj_args)
    node_coords_flattened = S.node_coords.flatten()
    sol = prb.run(x0=node_coords_flattened)
    curr_node_coords = sol.reshape(S.node_coords.shape)

    assert np.sum((true_curr_node_coords - curr_node_coords)**2) < 1e-6


def test_linear_elasticity_dual(setup_test):
    lc = 0.5
    L = 1.
    with pygmsh.geo.Geometry() as geom:
        p = geom.add_polygon([[0., 0.], [L, 0.], [L, L], [0., L]], mesh_size=lc)
        # create a default physical group for the boundary lines
        geom.add_physical(p.lines, label="boundary")
        geom.add_physical(p.lines[1], label="right")
        geom.add_physical(p.lines[3], label="left")
        mesh = geom.generate_mesh()

    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_DPP_weights()

    ref_node_coords = S.node_coords

    left_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "left")
    right_bnd_nodes_idx = util.get_nodes_for_physical_group(mesh, 1, "right")

    bottom_left_corner = left_bnd_nodes_idx.pop(0)

    mu_ = 1.
    lambda_ = 10.
    true_strain_xx = 0.02
    true_strain_yy = -(lambda_/(2*mu_+lambda_))*true_strain_xx
    true_curr_node_coords = S.node_coords.copy()
    true_curr_node_coords[:, 0] *= 1 + true_strain_xx
    true_curr_node_coords[:, 1] *= 1 + true_strain_yy
    left_bnd_pos_components = [0]
    right_bnd_pos_components = [0]

    left_bnd_nodes_pos = ref_node_coords[left_bnd_nodes_idx,
                                         :][:, left_bnd_pos_components]
    right_bnd_nodes_pos = true_curr_node_coords[right_bnd_nodes_idx,
                                                :][:, right_bnd_pos_components]
    bottom_left_corner_pos = ref_node_coords[bottom_left_corner, :]

    # NOTE: without flatten it does not work properly when concatenating multiple bcs;
    # fix this so that flatten is not needed (not intuitive)
    boundary_values = {"0": (left_bnd_nodes_idx + right_bnd_nodes_idx,
                             np.vstack((left_bnd_nodes_pos,
                                        right_bnd_nodes_pos)).flatten()),
                       ":": (bottom_left_corner, bottom_left_corner_pos)}

    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    gamma = 1000.

    embedded_dim = S.space_dim
    f = np.zeros((S.num_nodes, (embedded_dim-1))).flatten()

    obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values,
                'boundary_tractions': None}

    prb = optctrl.OptimizationProblem(dim=S.node_coords.size,
                                      state_dim=S.node_coords.size,
                                      objfun=ela.obj_linear_elasticity)

    prb.set_obj_args(obj_args)
    node_coords_flattened = S.node_coords.flatten()
    sol = prb.run(x0=node_coords_flattened)
    curr_node_coords = sol.reshape(S.node_coords.shape)

    print(ela.obj_linear_elasticity(
        true_curr_node_coords.flatten(), f, gamma, boundary_values, None))
    print(S.node_coords)

    assert np.sum((true_curr_node_coords - curr_node_coords)**2) < 1e-6
