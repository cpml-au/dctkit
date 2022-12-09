import numpy as np
from dctkit.mesh import simplex, util
from dctkit.math import shifted_list as sl
import os

cwd = os.path.dirname(simplex.__file__)


def test_boundary_COO():
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
    rows_index_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7])
    column_index_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3])
    values_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1])
    boundary_true = (rows_index_true, column_index_true, values_true)
    assert np.alltrue(boundary_tuple[0] == boundary_true[0])
    assert np.alltrue(boundary_tuple[1] == boundary_true[1])
    assert np.alltrue(boundary_tuple[2] == boundary_true[2])


def test_simplicial_complex():
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
    # define true boundary values
    boundary_true = sl.ShiftedList([], -1)
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 1, 5, 6, 3, 5, 7, 2, 4, 6, 7])
    values_1_true = np.array([-1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1])
    rows_2_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7])
    cols_2_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3])
    values_2_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1])
    boundary_true.append((rows_1_true, cols_1_true, values_1_true))
    boundary_true.append((rows_2_true, cols_2_true, values_2_true))

    # define true circumcenters

    circ_true = sl.ShiftedList([], -1)
    circ_1_true = np.array([[0.5, 0, 0], [1, 0.5, 0], [0.75, 0.25, 0], [0, 0.5, 0],
                           [0.25, 0.25, 0], [0.5, 1, 0], [0.75, 0.75, 0],
                           [0.25, 0.75, 0]])
    circ_2_true = np.array([[0.5, 0, 0], [1, 0.5, 0], [0, 0.5, 0], [0.5, 1, 0]])
    circ_true.append(circ_1_true)
    circ_true.append(circ_2_true)

    # define true primal volumes values
    pv_true = sl.ShiftedList([], -1)
    pv_1_true = np.array([1, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2, 1, np.sqrt(2)/2,
                          np.sqrt(2)/2])
    pv_2_true = np.array([0.25, 0.25, 0.25, 0.25])
    pv_true.append(pv_1_true)
    pv_true.append(pv_2_true)

    # define true dual volumes values
    dv_true = sl.ShiftedList([], -1)
    dv_1_true = np.array([0, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2, 0, np.sqrt(2)/2,
                          np.sqrt(2)/2])
    dv_2_true = np.array([1/8, 1/8, 1/8, 1/8, 1/2])
    dv_true.append(dv_1_true)
    dv_true.append(dv_2_true)

    # test boundary
    for i in range(3):
        assert np.alltrue(S.boundary[1][i] == boundary_true[1][i])
        assert np.alltrue(S.boundary[2][i] == boundary_true[2][i])

    # test circumcenters
    assert (np.linalg.norm(S.circ[1] - circ_true[1]) < 10**-8)
    assert (np.linalg.norm(S.circ[2] - circ_true[2]) < 10**-8)

    # test primal volumes
    assert (np.linalg.norm(S.primal_volumes[1] - pv_true[1]) < 10**-8)
    assert (np.linalg.norm(S.primal_volumes[2] - pv_true[2]) < 10**-8)

    # test dual volumes
    assert (np.linalg.norm(S.dual_volumes[1] - dv_true[1]) < 10**-8)
    assert (np.linalg.norm(S.dual_volumes[2] - dv_true[2]) < 10**-8)


if __name__ == "__main__":
    test_boundary_COO()
    test_simplicial_complex()
