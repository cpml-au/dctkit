import numpy as np
from dctkit.mesh import simplex, util
from dctkit.math import circumcenter
import os

cwd = os.path.dirname(simplex.__file__)


def test_circumcenter():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_p, x = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {S_p}")
    print(x)

    circ_true = np.empty((numElements, S_p.shape[1]))
    circ_true[0, :] = np.array([0.5, 0, 0])
    circ_true[1, :] = np.array([1, 0.5, 0])
    circ_true[2, :] = np.array([0, 0.5, 0])
    circ_true[3, :] = np.array([0.5, 1, 0])
    for i in range(numElements):
        assert np.allclose(circumcenter.circumcenter(S_p[i, :], x), circ_true[i, :])


if __name__ == '__main__':
    test_circumcenter()
