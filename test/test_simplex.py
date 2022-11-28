import numpy as np
from dctkit.mesh import simplex, util


def test_prova():
    numNodes, numElements, nodeTagsPerElem, x = util.read_mesh("test1.msh")
    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {nodeTagsPerElem}")
    # print(f"The coordinates of the nodes are \n {x}")
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


if __name__ == "__main__":
    test_prova()
