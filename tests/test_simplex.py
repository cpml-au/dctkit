import numpy as np
import dctkit
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl


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


def test_simplicial_complex_1(setup_test):
    num_nodes = 5
    mesh, _ = util.generate_line_mesh(num_nodes)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()

    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_true = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=dctkit.int_dtype)
    cols_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=dctkit.int_dtype)
    vals_true = np.array([-1,  1, -1,  1, -1,  1, -1,  1], dtype=dctkit.int_dtype)
    boundary_true.append((rows_true, cols_true, vals_true))

    # define true circumcenters
    circ_true = sl.ShiftedList([], -1)
    circ_true_1 = np.zeros((num_nodes - 1, 3))
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

    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])

    for i in range(2):
        assert np.allclose(S.circ[i], circ_true[i])
        assert np.allclose(S.primal_volumes[i], pv_true[i])
        assert np.allclose(S.dual_volumes[i], dv_true[i])
        assert np.allclose(S.hodge_star[i], hodge_true[i])
        assert np.allclose(S.hodge_star_inverse[i], hodge_inv_true[i])


def test_simplicial_complex_2(setup_test):
    mesh, _ = util.generate_square_mesh(1.0)
    S = util.build_complex_from_mesh(mesh, is_well_centered=False)
    S.get_hodge_star()
    S.get_flat_weights()

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

    # define true circumcenters

    circ_true = sl.ShiftedList([], -1)
    circ_1_true = np.array([[0.5, 0., 0.],
                            [0.,  0.5,  0.],
                            [0.25, 0.25, 0.],
                            [1.,   0.5,  0.],
                            [0.75, 0.25, 0.],
                            [0.5,  1.,   0.],
                            [0.75, 0.75, 0.],
                            [0.25, 0.75, 0.]], dtype=dctkit.float_dtype)
    circ_2_true = np.array([[0.5, 0.,  0.],
                            [0.,  0.5, 0.],
                            [1.,  0.5, 0.],
                            [0.5, 1.,  0.]],
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

    # define true dual edges
    dedges_true = np.array([[0., 0., 0.],
                            [0.,   0.,   0.],
                            [-0.5,  0.5,  0.],
                            [0.,   0.,   0.],
                            [-0.5, -0.5,  0.],
                            [0.,   0.,   0.],
                            [0.5, -0.5,  0.],
                            [0.5,  0.5,  0.]])

    # define true dual edges lengths
    num_n_simplices = S.S[S.dim].shape[0]
    num_nm1_simplices = S.S[S.dim-1].shape[0]
    dedges_lengths_true = np.zeros(
        (num_n_simplices, num_nm1_simplices), dtype=dctkit.float_dtype)
    dedges_lengths_true[0, [2, 4]] = np.sqrt(2)/4
    dedges_lengths_true[1, [2, 7]] = np.sqrt(2)/4
    dedges_lengths_true[2, [4, 6]] = np.sqrt(2)/4
    dedges_lengths_true[3, [6, 7]] = np.sqrt(2)/4

    metric_true = np.stack([np.identity(2)]*4)

    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])
        assert np.allclose(S.primal_volumes[i], pv_true[i])
        assert np.allclose(S.dual_volumes[i], dv_true[i])
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test circumcenters
    for i in range(1, 3):
        assert np.allclose(S.circ[i], circ_true[i])

    # test dual edge and dual edge lengths
    assert np.allclose(S.dual_edges_vectors, dedges_true)
    assert np.allclose(S.dual_edges_fractions_lengths, dedges_lengths_true)

    # test metric
    assert np.allclose(S.reference_metric, metric_true)

    # test hodge star inverse
    mesh, _ = util.generate_square_mesh(0.4)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()

    # FIXME: make this part of the test more clear (remove long instructions between
    # paretheses)
    n = S.dim
    for p in range(3):
        assert np.allclose(
            S.hodge_star[p]*S.hodge_star_inverse[p],
            (-1)**(p*(n-p))*np.ones(S.S[p].shape[0]))


def test_simplicial_complex_3(setup_test):
    # FIXME: generate mesh and complex after defining appropriate functions in util
    mesh, _ = util.generate_tet_mesh(2.0)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()

    # test boundary
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

    assert S.boundary[1][0].dtype == dctkit.int_dtype
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])
        assert np.alltrue(S.boundary[3][i] == boundary_true[3][i])

    # test circumcenter
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

    for i in range(1, 4):
        assert np.allclose(S.circ[i], circ_true[i])
    assert S.circ[1].dtype == dctkit.float_dtype

    # test primal volumes
    pv_true = sl.ShiftedList([], -1)
    pv_1_true = np.array([1., np.sqrt(5)/2, 1., np.sqrt(5)/2,
                         np.sqrt(2), 1.5], dtype=dctkit.float_dtype)
    pv_2_true = np.array([0.5, 0.5, 0.55901699, 0.75], dtype=dctkit.float_dtype)
    pv_3_true = np.array([5/30], dtype=dctkit.float_dtype)
    pv_true.append(pv_1_true)
    pv_true.append(pv_2_true)
    pv_true.append(pv_3_true)

    assert S.primal_volumes[1].dtype == dctkit.float_dtype
    for i in range(1, 4):
        assert np.allclose(S.primal_volumes[i], pv_true[i])

    # define true dual volumes values
    dv_true = []
    dv_0_true = np.array([0.0859375, 0.03255208, 0.02864583,
                         0.01953125], dtype=dctkit.float_dtype)
    dv_1_true = np.array([0.1875,  0.13975425,  0.171875,  0.03493856, -0.02209709,
                          -0.015625], dtype=dctkit.float_dtype)
    dv_2_true = np.array([0.5,  0.375,  0.2795085, -0.125], dtype=dctkit.float_dtype)
    dv_true.append(dv_0_true)
    dv_true.append(dv_1_true)
    dv_true.append(dv_2_true)

    assert S.dual_volumes[1].dtype == dctkit.float_dtype
    for i in range(3):
        assert np.allclose(S.dual_volumes[i], dv_true[i])

    # test hodge star
    hodge_true = [dv_0_true] + [dv_true[i]/pv_true[i]
                                for i in range(1, 3)] + [1/pv_3_true]
    for i in range(4):
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test hodge star inverse
    n = S.dim
    signed_identity = [S.hodge_star[i]*S.hodge_star_inverse[i] for i in range(4)]
    signed_identity_true = [(-1)**(i*(n-i))*np.ones(S.S[i].shape[0]) for i in range(4)]

    for i in range(4):
        assert np.allclose(signed_identity[i], signed_identity_true[i])
