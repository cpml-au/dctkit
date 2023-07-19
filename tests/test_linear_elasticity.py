import numpy as np
from dctkit.mesh import util
from dctkit.math.opt import optctrl
from dctkit.physics.elasticity import LinearElasticity
import dctkit as dt
import pygmsh
import pytest
import jax.numpy as jnp
import dctkit.dec.cochain as C
import dctkit.dec.vector as V

# cases = [[False, False], [True, False], [True, True]]
cases = [[False, False]]


@pytest.mark.parametrize('is_primal,energy_formulation', cases)
def test_linear_elasticity(setup_test, is_primal, energy_formulation):
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
    S.get_flat_DPP_weights()

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
    left_right_edges_idx = left_bnd_edges_idx+right_bnd_edges_idx
    bnd_tractions_free_values = np.zeros((len(idx_free_edges), 2), dtype=dt.float_dtype)
    bnd_tractions_left_right_values = np.zeros(
        (len(left_right_edges_idx)), dtype=dt.float_dtype)
    boundary_tractions = {'1': (left_right_edges_idx, bnd_tractions_left_right_values),
                          ':': (idx_free_edges, bnd_tractions_free_values)}

    ela = LinearElasticity(S=S, mu_=mu_, lambda_=lambda_)
    gamma = 1000.

    if energy_formulation:
        num_faces = S.S[2].shape[0]
        embedded_dim = S.space_dim
        f = np.zeros((num_faces, (embedded_dim-1))).flatten()
        obj = ela.obj_linear_elasticity_energy
        obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values}

    else:
        if is_primal:
            num_faces = S.S[2].shape[0]
            embedded_dim = S.space_dim
            f = np.zeros((num_faces, (embedded_dim-1))).flatten()
            obj = ela.obj_linear_elasticity_primal
        else:
            embedded_dim = S.space_dim
            f = np.zeros((S.num_nodes, (embedded_dim-1)),
                         dtype=dt.float_dtype).flatten()
            toy_matrix = jnp.zeros(S.node_coords.shape, dtype=dt.float_dtype)
            toy_matrix = toy_matrix.at[:].set(jnp.nan)
            obj = ela.obj_linear_elasticity_dual

        obj_args = {'f': f, 'gamma': gamma, 'boundary_values': boundary_values,
                    'boundary_tractions': boundary_tractions, 'toy_matrix': toy_matrix}

    # define x0
    node_coords_mod = S.node_coords.copy()
    node_coords_mod[left_bnd_nodes_idx + right_bnd_nodes_idx, 0] = np.nan
    node_coords_mod[bottom_left_corner] = np.nan
    node_coords_mod_flattened = node_coords_mod.flatten()
    x0 = node_coords_mod_flattened[~np.isnan(node_coords_mod_flattened)]

    prb = optctrl.OptimizationProblem(dim=len(x0),
                                      state_dim=len(x0),
                                      objfun=obj)

    '''
    node_coords_mod = true_curr_node_coords.copy()
    node_coords_mod[left_bnd_nodes_idx + right_bnd_nodes_idx, 0] = np.nan
    node_coords_mod[bottom_left_corner] = np.nan
    node_coords_mod_flattened = node_coords_mod.flatten()
    prova = node_coords_mod_flattened[~np.isnan(node_coords_mod_flattened)]

    print(true_curr_node_coords)

    print(obj(prova, f, gamma, boundary_values, boundary_tractions, toy_matrix))
    '''

    prb.set_obj_args(obj_args)
    sol = prb.run(x0=x0, ftol_abs=1e-9, ftol_rel=1e-9)

    # curr_node_coords = sol.reshape(S.node_coords.shape)
    # post-process solution
    for key in boundary_values:
        idx, values = boundary_values[key]
        if key == ":":
            toy_matrix = toy_matrix.at[idx, :].set(values)
        else:
            toy_matrix = toy_matrix.at[idx, int(key)].set(values)

    toy_matrix_flattened = toy_matrix.flatten()
    toy_matrix_flattened = toy_matrix_flattened.at[jnp.isnan(
        toy_matrix_flattened)].set(sol)
    curr_node_coords = toy_matrix_flattened.reshape(S.node_coords.shape)

    print(curr_node_coords)
    print(true_curr_node_coords)

    print("--------------------------------")
    strain = ela.get_GreenLagrange_strain(curr_node_coords)
    stress = ela.get_stress(strain)
    print(strain)
    print("--------------------------------")
    print(stress)

    node_coords_coch = C.CochainP0(complex=S, coeffs=curr_node_coords)
    f = f.reshape((S.num_nodes, S.space_dim-1))
    f_coch = C.CochainD2(complex=S, coeffs=f)
    print(ela.force_balance_residual_dual(
        node_coords_coch, f_coch, boundary_tractions).coeffs)
    print("--------------------------------------------")

    stress_tensor = V.DiscreteTensorFieldD(S=S, coeffs=stress.T, rank=2)
    # compute forces on dual edges
    stress_integrated = V.flat_DPD(stress_tensor)
    forces = C.star(stress_integrated)
    print(idx_free_edges)
    print(left_right_edges_idx)
    print(forces.coeffs)
    print("----------------------------------------")
    print(S.node_coords)
    print(S.S[1])

    assert np.allclose(true_curr_node_coords, curr_node_coords, atol=1e-3)
    assert False
