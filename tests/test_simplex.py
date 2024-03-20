import numpy as np
import dctkit
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl
import pytest

space_dim = [1, 2, 3]


def test_boundary_COO(setup_test):
    mesh, _ = util.generate_square_mesh(1.0)
    S2 = mesh.cells_dict["triangle"]

    boundary_tuple, _, _ = simplex.compute_boundary_COO(S2)

    rows_index_true = np.array(
        [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    column_index_true = np.array(
        [0, 1, 0, 1, 2, 0, 2, 3, 2, 3, 1, 3], dtype=dctkit.int_dtype)
    values_true = np.array(
        [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1], dtype=dctkit.int_dtype)

    assert boundary_tuple[0].dtype == dctkit.int_dtype
    boundary_true = (rows_index_true, column_index_true, values_true)
    assert np.alltrue(boundary_tuple[0] == boundary_true[0])
    assert np.alltrue(boundary_tuple[1] == boundary_true[1])
    assert np.alltrue(boundary_tuple[2] == boundary_true[2])


@pytest.mark.parametrize('space_dim', space_dim)
def test_simplicial_complex_1(setup_test, space_dim: int):
    num_nodes = 5
    space_dim = 1
    mesh, _ = util.generate_line_mesh(num_nodes)
    S = util.build_complex_from_mesh(mesh, space_dim=space_dim)
    S.get_hodge_star()
    S.get_primal_edge_vectors()
    S.get_complex_boundary_faces_indices()
    S.get_tets_containing_a_boundary_face()
    S.get_dual_edge_vectors()

    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_true = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=dctkit.int_dtype)
    cols_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=dctkit.int_dtype)
    vals_true = np.array([-1,  1, -1,  1, -1,  1, -1,  1], dtype=dctkit.int_dtype)
    boundary_true.append((rows_true, cols_true, vals_true))

    # define true boundary faces indices
    bnd_faces_indices_true = np.array([0, 4], dtype=dctkit.int_dtype)

    # define true tets idx containing a boundary face
    tets_cont_bnd_face_true = np.array([0, 3], dtype=dctkit.int_dtype).reshape(-1, 1)

    # define true circumcenters
    circ_true = sl.ShiftedList([], -1)
    circ_true_1 = np.zeros((num_nodes - 1, space_dim))
    circ_true_1[:, 0] = np.array([1/8, 3/8, 5/8, 7/8], dtype=dctkit.float_dtype)
    circ_true.append(circ_true_1)

    # define true primal volumes
    pv_true = [None]*2
    pv_true[0] = np.ones(num_nodes, dtype=dctkit.float_dtype)
    pv_true[1] = 1/4*np.ones(num_nodes-1, dtype=dctkit.float_dtype)

    # define true dual volumes values
    dv_true = [None]*2
    dv_true[0] = np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=dctkit.float_dtype)
    dv_true[1] = np.ones(4, dtype=dctkit.float_dtype)

    # define true hodge star values
    hodge_true = []
    hodge_true_0 = np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=dctkit.float_dtype)
    hodge_true_1 = np.array([4, 4, 4, 4], dtype=dctkit.float_dtype)
    hodge_true.append(hodge_true_0)
    hodge_true.append(hodge_true_1)

    # define true hodge star inverse values
    hodge_inv_true = []
    hodge_inv_true_0 = np.array(
        [8., 4., 4., 4., 8.], dtype=dctkit.float_dtype)
    hodge_inv_true_1 = np.array([0.25, 0.25, 0.25, 0.25], dtype=dctkit.float_dtype)
    hodge_inv_true.append(hodge_inv_true_0)
    hodge_inv_true.append(hodge_inv_true_1)

    # define true primal edge vectors
    primal_edges_true = np.zeros((num_nodes-1, space_dim), dtype=dctkit.float_dtype)
    primal_edges_true[:, 0] = 1/4*np.ones(S.S[1].shape[0])

    # define true dual edge vectors
    dual_edges_true = np.zeros(((num_nodes, space_dim)),  dtype=dctkit.float_dtype)
    dual_edges_true[1:-1, 0] = 0.25
    dual_edges_true[0, 0] = 0.125
    dual_edges_true[-1, 0] = 0.125

    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])

    for i in range(2):
        assert np.allclose(S.circ[i], circ_true[i])
        assert np.allclose(S.primal_volumes[i], pv_true[i])
        assert np.allclose(S.dual_volumes[i], dv_true[i])
        assert np.allclose(S.hodge_star[i], hodge_true[i])
        assert np.allclose(S.hodge_star_inverse[i], hodge_inv_true[i])

    assert np.allclose(S.bnd_faces_indices, bnd_faces_indices_true)
    assert np.allclose(S.tets_cont_bnd_face, tets_cont_bnd_face_true)
    assert np.allclose(S.primal_edges_vectors, primal_edges_true)
    assert np.allclose(S.dual_edges_vectors, dual_edges_true)


@pytest.mark.parametrize('space_dim', space_dim[1:])
def test_simplicial_complex_2(setup_test, space_dim):
    mesh, _ = util.generate_square_mesh(1.0)
    S = util.build_complex_from_mesh(mesh, is_well_centered=False, space_dim=space_dim)
    S.get_hodge_star()
    S.get_primal_edge_vectors()
    S.get_complex_boundary_faces_indices()
    S.get_tets_containing_a_boundary_face()
    S.get_dual_edge_vectors()
    S.get_flat_DPD_weights()
    S.get_flat_DPP_weights()
    S.get_flat_PDP_weights()
    num_edges = S.S[1].shape[0]

    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                           4, 4, 4, 4], dtype=dctkit.int_dtype)
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 3, 5, 6, 1, 5, 7, 2, 4, 6, 7],
                           dtype=dctkit.int_dtype)
    values_1_true = np.array([-1, -1, -1, 1, -1, -1, 1, -1, -1,
                             1, 1, -1, 1, 1, 1, 1], dtype=dctkit.int_dtype)
    rows_2_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    cols_2_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 2, 3, 1, 3],
                           dtype=dctkit.int_dtype)
    values_2_true = np.array([1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1],
                             dtype=dctkit.int_dtype)
    boundary_true.append((rows_1_true, cols_1_true, values_1_true))
    boundary_true.append((rows_2_true, cols_2_true, values_2_true))

    # define true bnd faces indices
    bnd_faces_indices_true = np.array([0, 1, 3, 5], dtype=dctkit.int_dtype)

    # define true tets containing bnd face
    tets_cont_bnd_face_true = np.arange(4, dtype=dctkit.int_dtype).reshape(-1, 1)

    # define true circumcenters
    circ_true = sl.ShiftedList([], -1)
    circ_1_true = np.zeros((num_edges, space_dim), dtype=dctkit.float_dtype)
    circ_1_true[:, :2] = np.array([[0.5, 0.],
                                   [0.,  0.5],
                                   [0.25, 0.25],
                                   [1.,   0.5],
                                   [0.75, 0.25],
                                   [0.5,  1.],
                                   [0.75, 0.75],
                                   [0.25, 0.75]])
    circ_2_true = np.zeros((S.S[2].shape[0], space_dim), dtype=dctkit.float_dtype)
    circ_2_true[:, :2] = np.array([[0.5, 0.],
                                   [0.,  0.5],
                                   [1.,  0.5],
                                   [0.5, 1.]],
                                  dtype=dctkit.float_dtype)
    circ_true.append(circ_1_true)
    circ_true.append(circ_2_true)

    # define true primal volumes values
    pv_true = [None]*3
    pv_1_true = np.array([1, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2,
                          np.sqrt(2)/2], dtype=dctkit.float_dtype)
    pv_2_true = np.array([0.25, 0.25, 0.25, 0.25], dtype=dctkit.float_dtype)
    pv_true[0] = np.ones(S.num_nodes, dtype=dctkit.float_dtype)
    pv_true[1] = pv_1_true
    pv_true[2] = pv_2_true

    # define true dual volumes values
    dv_true = [None]*3
    dv_1_true = np.array([1/8, 1/8, 1/8, 1/8, 1/2], dtype=dctkit.float_dtype)
    dv_2_true = np.array([0, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2,
                          np.sqrt(2)/2], dtype=dctkit.float_dtype)
    dv_true[0] = dv_1_true
    dv_true[1] = dv_2_true
    dv_true[2] = np.ones(S.S[2].shape[0], dtype=dctkit.float_dtype)

    # define true hodge star values
    hodge_true = []
    hodge_0_true = np.array([1/8, 1/8, 1/8, 1/8, 1/2], dtype=dctkit.float_dtype)
    hodge_1_true = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=dctkit.float_dtype)
    hodge_2_true = np.array([4, 4, 4, 4], dtype=dctkit.float_dtype)
    hodge_true.append(hodge_0_true)
    hodge_true.append(hodge_1_true)
    hodge_true.append(hodge_2_true)

    # define true primal edge vectors
    primal_edges_vectors_true = np.zeros(
        (num_edges, space_dim), dtype=dctkit.float_dtype)
    primal_edges_vectors_true[:, :2] = np.array([[1.,  0.],
                                                [0.,  1.],
                                                 [0.5,  0.5],
                                                 [0.,  1.],
                                                 [-0.5,  0.5],
                                                 [-1.,  0.],
                                                 [-0.5, -0.5],
                                                 [0.5, -0.5]])

    # define true dual edge vectors
    dual_edges_vectors_true = np.zeros(
        (num_edges, space_dim), dtype=dctkit.float_dtype)
    dual_edges_vectors_true[:, :2] = np.array([[0., 0.],
                                               [0.,   0.],
                                               [-0.5,  0.5],
                                               [0.,   0.],
                                               [-0.5, -0.5],
                                               [0.,   0.],
                                               [0.5, -0.5],
                                               [0.5,  0.5]])

    # define true dual edges lengths
    num_n_simplices = S.S[S.dim].shape[0]
    num_nm1_simplices = S.S[S.dim-1].shape[0]
    dedges_lengths_true = np.zeros(
        (num_n_simplices, num_nm1_simplices), dtype=dctkit.float_dtype)
    dedges_lengths_true[0, [2, 4]] = np.sqrt(2)/4
    dedges_lengths_true[1, [2, 7]] = np.sqrt(2)/4
    dedges_lengths_true[2, [4, 6]] = np.sqrt(2)/4
    dedges_lengths_true[3, [6, 7]] = np.sqrt(2)/4

    # define true flat DPD and DPP matrices
    flat_DPD_weights_true = np.array([[0., 0., 0.5, 0., 0.5, 0., 0., 0.],
                                      [0., 0., 0.5, 0., 0., 0., 0., 0.5],
                                      [0., 0., 0., 0., 0.5, 0., 0.5, 0.],
                                      [0., 0., 0., 0., 0., 0., 0.5, 0.5]],
                                     dtype=dctkit.float_dtype)
    flat_DPP_weights_true = flat_DPD_weights_true
    flat_PDP_weights_true = np.array([[0.5, 0.5, 0., 0., 0.],
                                      [0.5, 0., 0., 0.5, 0.],
                                      [0.35355339, 0., 0., 0., 0.35355339],
                                      [0., 0.5, 0.5, 0., 0.],
                                      [0., 0.35355339, 0., 0., 0.35355339],
                                      [0., 0., 0.5, 0.5, 0.],
                                      [0., 0., 0.35355339, 0., 0.35355339],
                                      [0., 0., 0., 0.35355339, 0.35355339]],
                                     dtype=dctkit.float_dtype)

    # define true reference metric
    metric_true = np.stack([np.identity(2)]*4)

    # test boundary
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])
        assert np.allclose(S.primal_volumes[i], pv_true[i])
        assert np.allclose(S.dual_volumes[i], dv_true[i])
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test bnd faces indices
    assert np.allclose(S.bnd_faces_indices, bnd_faces_indices_true)

    # test tets containing boundary face
    assert np.allclose(S.tets_cont_bnd_face, tets_cont_bnd_face_true)

    # test circumcenters
    for i in range(1, 3):
        assert np.allclose(S.circ[i], circ_true[i])

    # test primal edge
    assert np.allclose(S.primal_edges_vectors, primal_edges_vectors_true)

    # test dual edge and dual edge lengths
    assert np.allclose(S.dual_edges_vectors, dual_edges_vectors_true)
    assert np.allclose(S.dual_edges_fractions_lengths, dedges_lengths_true)

    # test flat DPD
    assert np.allclose(S.flat_DPD_weights, flat_DPD_weights_true)

    # test flat DPP
    # FIXME: after extending flat DPP to other dimensions, test it!
    assert np.allclose(S.flat_DPP_weights, flat_DPP_weights_true)

    # test flat PDP
    assert np.allclose(S.flat_PDP_weights, flat_PDP_weights_true)

    # test metric
    assert np.allclose(S.reference_metric, metric_true)

    # test hodge star inverse
    mesh, _ = util.generate_square_mesh(0.4)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()

    n = S.dim
    signed_identity = [S.hodge_star[i]*S.hodge_star_inverse[i] for i in range(3)]
    signed_identity_true = [(-1)**(i*(n-i))*np.ones(S.S[i].shape[0]) for i in range(3)]
    for i in range(3):
        assert np.allclose(signed_identity[i], signed_identity_true[i])


@pytest.mark.parametrize('space_dim', space_dim[2:])
def test_simplicial_complex_3(setup_test, space_dim):
    # FIXME: generate mesh and complex after defining appropriate functions in util
    mesh, _ = util.generate_tet_mesh(2.0)
    S = util.build_complex_from_mesh(mesh, space_dim)
    S.get_hodge_star()
    S.get_primal_edge_vectors()
    S.get_complex_boundary_faces_indices()
    S.get_tets_containing_a_boundary_face()
    S.get_dual_edge_vectors()

    # define true boundary
    boundary_true = sl.ShiftedList([], -1)
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=dctkit.int_dtype)
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 1, 3, 5, 2, 4, 5], dtype=dctkit.int_dtype)
    values_1_true = np.array([-1, -1, -1,  1, -1, -1,  1,  1, -1,
                             1,  1,  1], dtype=dctkit.int_dtype)
    rows_2_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=dctkit.int_dtype)
    cols_2_true = np.array([0, 1, 0, 2, 1, 2, 0, 3, 1, 3, 2, 3], dtype=dctkit.int_dtype)
    values_2_true = np.array([1,  1, -1,  1, -1, -1,  1,  1,  1, -
                             1,  1,  1], dtype=dctkit.int_dtype)
    rows_3_true = np.array([0, 1, 2, 3], dtype=dctkit.int_dtype)
    cols_3_true = np.array([0, 0, 0, 0], dtype=dctkit.int_dtype)
    values_3_true = np.array([-1,  1, -1,  1], dtype=dctkit.int_dtype)
    boundary_true.append((rows_1_true, cols_1_true, values_1_true))
    boundary_true.append((rows_2_true, cols_2_true, values_2_true))
    boundary_true.append((rows_3_true, cols_3_true, values_3_true))

    # define true boundary faces indices
    bnd_faces_indices_true = np.arange(4, dtype=dctkit.int_dtype)

    # define true tets idx containing a boundary face
    tets_cont_bnd_face_true = np.zeros(4, dtype=dctkit.int_dtype).reshape(-1, 1)

    # define true circumcenter
    circ_true = sl.ShiftedList([], -1)
    circ_1_true = np.array([[0.5, 0., 0.],
                            [0.25, 0.5, 0.],
                            [0., 0., 0.5],
                            [0.75, 0.5, 0.],
                            [0.5, 0., 0.5],
                            [0.25, 0.5, 0.5]], dtype=dctkit.float_dtype)
    circ_2_true = np.array([[0.5, 0.375, 0.],
                            [0.5, 0., 0.5],
                            [0.25, 0.5, 0.5],
                            [0.41666667, 0.33333333, 0.41666667]],
                           dtype=dctkit.float_dtype)
    circ_3_true = np.array([[0.5, 0.375, 0.5]], dtype=dctkit.float_dtype)
    circ_true.append(circ_1_true)
    circ_true.append(circ_2_true)
    circ_true.append(circ_3_true)

    # define true primal volumes
    pv_true = sl.ShiftedList([], -1)
    pv_1_true = np.array([1., np.sqrt(5)/2, 1., np.sqrt(5)/2,
                         np.sqrt(2), 1.5], dtype=dctkit.float_dtype)
    pv_2_true = np.array([0.5, 0.5, 0.55901699, 0.75], dtype=dctkit.float_dtype)
    pv_3_true = np.array([5/30], dtype=dctkit.float_dtype)
    pv_true.append(pv_1_true)
    pv_true.append(pv_2_true)
    pv_true.append(pv_3_true)

    # define true dual volumes
    dv_true = []
    dv_0_true = np.array([0.0859375, 0.03255208, 0.02864583,
                         0.01953125], dtype=dctkit.float_dtype)
    dv_1_true = np.array([0.1875,  0.13975425,  0.171875,  0.03493856, -0.02209709,
                          -0.015625], dtype=dctkit.float_dtype)
    dv_2_true = np.array([0.5,  0.375,  0.2795085, -0.125], dtype=dctkit.float_dtype)
    dv_true.append(dv_0_true)
    dv_true.append(dv_1_true)
    dv_true.append(dv_2_true)

    # define true hodge star
    hodge_true = [dv_0_true] + [dv_true[i]/pv_true[i]
                                for i in range(1, 3)] + [1/pv_3_true]

    # define true hodge star inverse
    n = S.dim
    signed_identity = [S.hodge_star[i]*S.hodge_star_inverse[i] for i in range(4)]
    signed_identity_true = [(-1)**(i*(n-i))*np.ones(S.S[i].shape[0]) for i in range(4)]

    # define true primal edges vectors
    primal_edges_vectors_true = np.array([[1.,  0.,  0.],
                                          [0.5,  1.,  0.],
                                          [0.,  0.,  1.],
                                          [-0.5,  1.,  0.],
                                          [-1.,  0.,  1.],
                                          [-0.5, -1.,  1.]], dtype=dctkit.float_dtype)

    # define true dual edges vectors
    dual_edges_vectors_true = np.array([[0.,  0.,  0.5],
                                        [0., -0.375,  0.],
                                        [0.25, -0.125,  0.],
                                        [-0.08333333, -0.04166667, -0.08333333]],
                                       dtype=dctkit.float_dtype)

    # test boundary
    assert S.boundary[1][0].dtype == dctkit.int_dtype
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])
        assert np.alltrue(S.boundary[3][i] == boundary_true[3][i])

    # test bnd faces indices
    assert np.allclose(S.bnd_faces_indices, bnd_faces_indices_true)

    # test tets containing boundary face
    assert np.allclose(S.tets_cont_bnd_face, tets_cont_bnd_face_true)

    # test circumcenter
    for i in range(1, 4):
        assert np.allclose(S.circ[i], circ_true[i])
    assert S.circ[1].dtype == dctkit.float_dtype

    # test primal volumes
    assert S.primal_volumes[1].dtype == dctkit.float_dtype
    for i in range(1, 4):
        assert np.allclose(S.primal_volumes[i], pv_true[i])

    # test dual volumes
    assert S.dual_volumes[1].dtype == dctkit.float_dtype
    for i in range(3):
        assert np.allclose(S.dual_volumes[i], dv_true[i])

    # test hodge star and hodge star inverse
    for i in range(4):
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    for i in range(4):
        assert np.allclose(signed_identity[i], signed_identity_true[i])

    # test primal edge vectors
    assert np.allclose(S.primal_edges_vectors, primal_edges_vectors_true)

    # test dual edge vectors
    assert np.allclose(S.dual_edges_vectors, dual_edges_vectors_true)
