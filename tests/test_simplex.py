import numpy as np
from src.dctkit.mesh import simplex, util
import os

cwd = os.path.dirname(simplex.__file__)


def test_compute_face_to_edge_connectivity():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, nodeTagsPerElem, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {nodeTagsPerElem}")

    C, NtE, EtF = simplex.compute_face_to_edge_connectivity(nodeTagsPerElem)
    print(f"The orientation matrix is \n {C}")
    print(f"The NtE matrix is \n {NtE}")
    print(f"The EtF matrix is \n {EtF}")

    C_true = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1], [-1, 1, -1]])
    EtF_true = np.array([[0, 1, 2], [3, 1, 5], [6, 0, 8], [6, 3, 11]])
    NtE_true = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3],
                         [2, 4], [3, 4]])

    assert np.alltrue(C == C_true)
    assert np.alltrue(EtF == EtF_true)
    assert np.alltrue(NtE == NtE_true)


def test_boundary_COO():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The face matrix is \n {S_2}")

    boundary_tuple, _ = simplex.compute_boundary_COO(
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
    numNodes, numElements, S_2, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The face matrix is \n {S_2}")

    S = simplex.SimComplex(S_2)
    boundary = S.get_boundary_operators()
    boundary_true = []
    rows_1_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    cols_1_true = np.array([0, 1, 2, 0, 3, 4, 1, 5, 6, 3, 5, 7, 2, 4, 6, 7])
    values_1_true = np.array([-1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1])
    rows_2_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7])
    cols_2_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3])
    values_2_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1])
    boundary_true.append((rows_1_true, cols_1_true, values_1_true))
    boundary_true.append((rows_2_true, cols_2_true, values_2_true))
    for i in range(3):
        assert np.alltrue(boundary[0][i] == boundary_true[0][i])
        assert np.alltrue(boundary[1][i] == boundary_true[1][i])


if __name__ == "__main__":
    test_simplicial_complex()
