import numpy as np
import dctkit
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl
import os
import matplotlib.tri as tri
import matplotlib.pyplot as plt


cwd = os.path.dirname(simplex.__file__)


def test_boundary_COO(int_dtype=dctkit.IntDtype.int32):
    dctkit.int_dtype = int_dtype.name
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, _ = util.read_mesh(full_path)

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


def test_simplicial_complex(float_dtype=dctkit.FloatDtype.float64, int_dtype=dctkit.IntDtype.int64):
    dctkit.float_dtype = float_dtype.name
    dctkit.int_dtype = int_dtype.name
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, x = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The face matrix is \n {S_2}")

    S = simplex.SimplicialComplex(S_2, x)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    print(S.boundary)
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

    # test hodge star inverse
    _, _, S_2_new, node_coords_new = util.generate_square_mesh(0.4)
    triang = tri.Triangulation(node_coords_new[:, 0], node_coords_new[:, 1])

    plt.triplot(triang, 'ko-')
    plt.show()

    # FIXME: make this part of the test more clear (remove long instructions between paretheses)
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


if __name__ == "__main__":
    test_boundary_COO(dctkit.IntDtype.int32)
    test_boundary_COO(dctkit.IntDtype.int64)
    test_simplicial_complex(dctkit.FloatDtype.float32, dctkit.IntDtype.int32)
    test_simplicial_complex(dctkit.FloatDtype.float64, dctkit.IntDtype.int64)
