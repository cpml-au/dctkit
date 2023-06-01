import numpy as np
import dctkit
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl
import os

cwd = os.path.dirname(__file__)


def test_boundary_COO(setup_test):
    filename = "data/test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, _, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The face matrix is \n {S_2}")

    boundary_tuple, _, _ = simplex.compute_boundary_COO(
        S_2)
    print(f"The row index vector is \n {boundary_tuple[0]}")
    print(f"The column index vector is \n {boundary_tuple[1]}")
    print(f"The values vector is \n {boundary_tuple[2]}")
    rows_index_true = np.array(
        [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    column_index_true = np.array(
        [0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3], dtype=dctkit.int_dtype)
    values_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1,
                           1, 1, -1], dtype=dctkit.int_dtype)

    assert boundary_tuple[0].dtype == dctkit.int_dtype
    boundary_true = (rows_index_true, column_index_true, values_true)
    assert np.alltrue(boundary_tuple[0] == boundary_true[0])
    assert np.alltrue(boundary_tuple[1] == boundary_true[1])
    assert np.alltrue(boundary_tuple[2] == boundary_true[2])


def test_simplicial_complex_1(setup_test):
    # define node_coords
    num_nodes = 5
    node_coords = np.linspace(0, 1, num=num_nodes)
    x = np.zeros((num_nodes, 2))
    x[:, 0] = node_coords
    # define S_1
    S_1 = np.empty((num_nodes - 1, 2))
    S_1[:, 0] = np.arange(num_nodes-1)
    S_1[:, 1] = np.arange(1, num_nodes)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_true = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=dctkit.int_dtype)
    cols_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=dctkit.int_dtype)
    vals_true = np.array([-1,  1, -1,  1, -1,  1, -1,  1], dtype=dctkit.int_dtype)
    boundary_true.append((rows_true, cols_true, vals_true))

    # define true circumcenters
    circ_true = sl.ShiftedList([], -1)
    circ_true_1 = np.zeros((num_nodes - 1, 2))
    circ_true_1[:, 0] = np.array([1/8, 3/8, 5/8, 7/8], dtype=dctkit.float_dtype)
    circ_true.append(circ_true_1)

    # define true primal volumes
    pv_true = sl.ShiftedList([], -1)
    pv_true.append(1/4*np.ones(num_nodes-1, dtype=dctkit.float_dtype))

    # define true dual volumes values
    dv_true = sl.ShiftedList([], -1)
    dv_true.append(np.array([0.125, 0.25, 0.25, 0.25, 0.125], dtype=dctkit.float_dtype))

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

    # test boundary
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])

    # test circumcenters
    assert np.allclose(S.circ[1], circ_true[1])

    # test primal volumes
    assert np.allclose(S.primal_volumes[1], pv_true[1])

    # test dual volumes
    assert np.allclose(S.dual_volumes[1], dv_true[1])

    # test hodge star
    for i in range(2):
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test hodge star inverse
    for i in range(2):
        assert np.allclose(S.hodge_star_inverse[i], hodge_inv_true[i])


def test_simplicial_complex_2(setup_test):
    filename = "data/test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, x, belem = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The face matrix is \n {S_2}")

    S = simplex.SimplicialComplex(S_2, x, belem=belem)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    S.get_dual_elements()
    S.get_dual_relative_areas()
    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                           4, 4, 4, 4], dtype=dctkit.int_dtype)
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 1, 5, 6, 3, 5, 7,
                           2, 4, 6, 7], dtype=dctkit.int_dtype)
    values_1_true = np.array([-1, -1, -1, 1, -1, -1, 1, -1, -1,
                             1, 1, -1, 1, 1, 1, 1], dtype=dctkit.int_dtype)
    rows_2_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    cols_2_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3], dtype=dctkit.int_dtype)
    values_2_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1,
                             1, 1, -1], dtype=dctkit.int_dtype)
    boundary_true.append((rows_1_true, cols_1_true, values_1_true))
    boundary_true.append((rows_2_true, cols_2_true, values_2_true))

    # define true circumcenters

    circ_true = sl.ShiftedList([], -1)
    circ_1_true = np.array([[0.5, 0, 0], [1, 0.5, 0], [0.75, 0.25, 0], [0, 0.5, 0],
                           [0.25, 0.25, 0], [0.5, 1, 0], [0.75, 0.75, 0],
                           [0.25, 0.75, 0]], dtype=dctkit.float_dtype)
    circ_2_true = np.array([[0.5, 0, 0], [1, 0.5, 0], [0, 0.5, 0], [
                           0.5, 1, 0]], dtype=dctkit.float_dtype)
    circ_true.append(circ_1_true)
    circ_true.append(circ_2_true)

    # define true primal volumes values
    pv_true = sl.ShiftedList([], -1)
    pv_1_true = np.array([1, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2,
                          np.sqrt(2)/2], dtype=dctkit.float_dtype)
    pv_2_true = np.array([0.25, 0.25, 0.25, 0.25], dtype=dctkit.float_dtype)
    pv_true.append(pv_1_true)
    pv_true.append(pv_2_true)

    # define true dual volumes values
    dv_true = sl.ShiftedList([], -1)
    dv_1_true = np.array([0, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2,
                          np.sqrt(2)/2], dtype=dctkit.float_dtype)
    dv_2_true = np.array([1/8, 1/8, 1/8, 1/8, 1/2], dtype=dctkit.float_dtype)
    dv_true.append(dv_1_true)
    dv_true.append(dv_2_true)

    # define true hodge star values
    hodge_true = []
    hodge_0_true = np.array([1/8, 1/8, 1/8, 1/8, 1/2], dtype=dctkit.float_dtype)
    hodge_1_true = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=dctkit.float_dtype)
    hodge_2_true = np.array([4, 4, 4, 4], dtype=dctkit.float_dtype)
    hodge_true.append(hodge_0_true)
    hodge_true.append(hodge_1_true)
    hodge_true.append(hodge_2_true)

    # define true dual edges
    delements_true = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0],
                               [0, 0, 0], [0.5, -0.5, 0], [0, 0, 0],
                               [-0.5, 0.5, 0], [-0.5, -0.5, 0]])

    # define true dual edges lengths
    num_n_simplices = S.S[S.dim].shape[0]
    num_nm1_simplices = S.S[S.dim-1].shape[0]
    delements_lengths_true = -np.ones(
        (num_n_simplices, num_nm1_simplices), dtype=dctkit.float_dtype)
    delements_lengths_true[0, 0] = 0
    delements_lengths_true[0, [2, 4]] = np.sqrt(2)/4
    delements_lengths_true[1, 1] = 0
    delements_lengths_true[1, [2, 6]] = np.sqrt(2)/4
    delements_lengths_true[2, 3] = 0
    delements_lengths_true[2, [4, 7]] = np.sqrt(2)/4
    delements_lengths_true[3, 5] = 0
    delements_lengths_true[3, [6, 7]] = np.sqrt(2)/4

    assert S.boundary[1][0].dtype == dctkit.int_dtype
    assert S.circ[1].dtype == dctkit.float_dtype
    assert S.primal_volumes[1].dtype == dctkit.float_dtype
    assert S.dual_volumes[1].dtype == dctkit.float_dtype
    assert S.hodge_star[0].dtype == dctkit.float_dtype

    # test boundary
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])

    # test circumcenters
    assert np.allclose(S.circ[1], circ_true[1])
    assert np.allclose(S.circ[2], circ_true[2])

    # test primal volumes
    assert np.allclose(S.primal_volumes[1], pv_true[1])
    assert np.allclose(S.primal_volumes[2], pv_true[2])

    # test dual volumes
    assert np.allclose(S.dual_volumes[1], dv_true[1])
    assert np.allclose(S.dual_volumes[2], dv_true[2])

    # test hodge star
    for i in range(3):
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test dual edge and dual edge lengths
    assert np.allclose(S.delements, delements_true)
    assert np.allclose(S.delements_areas, delements_lengths_true)

    # test hodge star inverse
    _, _, S_2_new, node_coords_new, _ = util.generate_square_mesh(0.4)

    # FIXME: make this part of the test more clear (remove long instructions between
    # paretheses)
    cpx_new = simplex.SimplicialComplex(S_2_new, node_coords_new, is_well_centered=True)
    cpx_new.get_circumcenters()
    cpx_new.get_primal_volumes()
    cpx_new.get_dual_volumes()
    cpx_new.get_hodge_star()
    n = cpx_new. dim
    for p in range(3):
        assert np.allclose(
            cpx_new.hodge_star[p]*cpx_new.hodge_star_inverse[p], (-1)**(
                p*(n-p))*np.ones(cpx_new.S[p].shape[0]))
