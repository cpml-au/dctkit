import numpy as np
import dctkit
from dctkit import FloatDtype, IntDtype
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl
import os
import matplotlib.tri as tri
import pytest
from jax.experimental import sparse

cwd = os.path.dirname(__file__)


@pytest.mark.parametrize('int_dtype', [IntDtype.int32, IntDtype.int64])
def test_boundary_COO(int_dtype):
    dctkit.int_dtype = int_dtype.name
    filename = "data/test1.msh"
    full_path = os.path.join(cwd, filename)
    _, _, S_2, _ = util.read_mesh(full_path)
    boundary, _, _ = simplex.compute_boundary_COO(S_2)
    rows_index_true = np.array(
        [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    column_index_true = np.array(
        [0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3], dtype=dctkit.int_dtype)
    values_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1,
                           1, 1, -1], dtype=dctkit.float_dtype)
    indices_true = np.column_stack([rows_index_true, column_index_true])

    # NOTE: mesh1 has 8 edges
    boundary_true = sparse.BCOO([values_true, indices_true], shape=(8, S_2.shape[0]))
    assert np.allclose(boundary.todense(), boundary_true.todense())


@pytest.mark.parametrize('float_dtype,int_dtype', [[FloatDtype.float32,
                                                    IntDtype.int32],
                                                   [FloatDtype.float64,
                                                    IntDtype.int64]])
def test_simplicial_complex_1(float_dtype, int_dtype):
    dctkit.float_dtype = float_dtype.name
    dctkit.int_dtype = int_dtype.name
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

    # define true boundary operator from edges to nodes
    # boundary_true = sl.ShiftedList([], -1)
    rows_true = np.array([0, 1, 1, 2, 2, 3, 3, 4], dtype=dctkit.int_dtype)
    cols_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=dctkit.int_dtype)
    vals_true = np.array([-1,  1, -1,  1, -1,  1, -1,  1], dtype=dctkit.float_dtype)
    indices_true = np.column_stack([rows_true, cols_true])
    boundary_true = sparse.BCOO([vals_true, indices_true],
                                shape=(num_nodes, num_nodes-1))

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
    assert np.allclose(S.boundary[1].todense(), boundary_true.todense())

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


@pytest.mark.parametrize('float_dtype,int_dtype', [[FloatDtype.float32,
                                                    IntDtype.int32],
                                                   [FloatDtype.float64,
                                                    IntDtype.int64]])
def test_simplicial_complex_2(float_dtype, int_dtype):
    dctkit.float_dtype = float_dtype.name
    dctkit.int_dtype = int_dtype.name
    filename = "data/test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, x = util.read_mesh(full_path)

    S = simplex.SimplicialComplex(S_2, x)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # define true boundary matrices
    boundary_true = sl.ShiftedList([None]*2, -1)
    # boundary_true = [None]*3
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                           4, 4, 4, 4], dtype=dctkit.int_dtype)
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 1, 5, 6, 3, 5, 7,
                           2, 4, 6, 7], dtype=dctkit.int_dtype)
    values_1_true = np.array([-1, -1, -1, 1, -1, -1, 1, -1, -1,
                             1, 1, -1, 1, 1, 1, 1], dtype=dctkit.float_dtype)
    indices_1_true = np.column_stack([rows_1_true, cols_1_true])
    rows_2_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7], dtype=dctkit.int_dtype)
    cols_2_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3], dtype=dctkit.int_dtype)
    values_2_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1,
                             1, 1, -1], dtype=dctkit.float_dtype)
    indices_2_true = np.column_stack([rows_2_true, cols_2_true])
    boundary_true[1] = sparse.BCOO([values_1_true, indices_1_true], shape=(numNodes, 8))
    boundary_true[2] = sparse.BCOO(
        [values_2_true, indices_2_true], shape=(8, numElements))

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

    assert S.circ[1].dtype == dctkit.float_dtype
    assert S.primal_volumes[1].dtype == dctkit.float_dtype
    assert S.dual_volumes[1].dtype == dctkit.float_dtype
    assert S.hodge_star[0].dtype == dctkit.float_dtype

    # test boundary, circumcenters, primal volumes and dual volumes
    for i in range(2):
        assert np.allclose(S.boundary[i].todense(), boundary_true[i].todense())
        assert np.allclose(S.circ[i], circ_true[i])
        assert np.allclose(S.primal_volumes[i], pv_true[i])
        assert np.allclose(S.dual_volumes[i], dv_true[i])

    # test hodge star
    for i in range(3):
        assert np.allclose(S.hodge_star[i], hodge_true[i])

    # test hodge star inverse
    _, _, S_2_new, node_coords_new = util.generate_square_mesh(0.4)
    triang = tri.Triangulation(node_coords_new[:, 0], node_coords_new[:, 1])

    # plt.triplot(triang, 'ko-')
    # plt.show()

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
