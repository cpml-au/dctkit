import numpy as np
from dctkit.mesh import simplex, util
from dctkit.math import circumcenter as circ
import os

cwd = os.path.dirname(simplex.__file__)


def test_circumcenter():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {S_2}")
    print(f"The coordinates of the nodes are \n {node_coord}")

    circ_true = np.empty((numElements, S_2.shape[1]))
    circ_true[0, :] = np.array([0.5, 0, 0])
    circ_true[1, :] = np.array([1, 0.5, 0])
    circ_true[2, :] = np.array([0, 0.5, 0])
    circ_true[3, :] = np.array([0.5, 1, 0])
    for i in range(numElements):
        assert np.allclose(circ.circumcenter(S_2[i, :], node_coord)[0], circ_true[i, :])


def test_dual_volumes():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

    K = simplex.SimplicialComplex(S_2, node_coord)
    K.boundary_simplices[1] = np.array([[0, 2, 4], [1, 2, 6], [3, 4, 7], [5, 6, 7]],
                                       dtype=np.int32)
    K.boundary_simplices[0] = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 4], [2, 3],
                                        [2, 4], [3, 4]], dtype=np.int32)
    print(K.circ[0], K.circ[1])
    K.get_dual_volumes()


if __name__ == '__main__':
    test_dual_volumes()
    test_circumcenter()
