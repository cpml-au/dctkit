import numpy as np
from dctkit.mesh import simplex, util
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
    numNodes, numElements, nodeTagsPerElem, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {nodeTagsPerElem}")

    rows_index, column_index, values = simplex.compute_boundary_COO(
        nodeTagsPerElem)
    print(f"The row index vector is \n {rows_index}")
    print(f"The column index vector is \n {column_index}")
    print(f"The values vector is \n {values}")

    rows_index_true = np.array([0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7])
    column_index_true = np.array([0, 1, 0, 1, 2, 0, 2, 3, 1, 3, 2, 3])
    values_true = np.array([1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1])
    assert np.alltrue(rows_index == rows_index_true)
    assert np.alltrue(column_index == column_index_true)
    assert np.alltrue(values == values_true)


if __name__ == "__main__":
    test_boundary_COO()
