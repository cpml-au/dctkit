import numpy as np
from dctkit.mesh import simplex, util
from dctkit.dec import cochain
import os

cwd = os.path.dirname(simplex.__file__)


def test_cochain():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, _ = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {S_2}")

    v_0 = np.array([1, 2, 3, 4, 5])
    v_0_true = np.array([1, 2, 4, 2, 3, 1, 2, 1])
    v_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    v_1_true = np.array([3, -6, 7, -7])
    c_0 = cochain.Cochain(dim=0, is_primal=True, node_tags=S_2, vec=v_0)
    c_1 = cochain.Cochain(dim=1, is_primal=True, node_tags=S_2, vec=v_1)
    dc_0 = cochain.coboundary_operator(c_0)
    dc_1 = cochain.coboundary_operator(c_1)
    dc_0_true = cochain.Cochain(dim=0, is_primal=True, node_tags=S_2, vec=v_0_true)
    dc_1_true = cochain.Cochain(dim=1, is_primal=True, node_tags=S_2, vec=v_1_true)
    assert np.alltrue(dc_0.vec == dc_0_true.vec)
    assert np.alltrue(dc_1.vec == dc_1_true.vec)


if __name__ == '__main__':
    test_cochain()
