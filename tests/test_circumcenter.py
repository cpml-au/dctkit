import numpy as np
import dctkit
from dctkit.mesh import util
from dctkit.mesh import circumcenter as circ
import os

cwd = os.path.dirname(__file__)


def test_circumcenter(setup_test):
    mesh, _ = util.generate_square_mesh(1.0)
    S2 = mesh.cells_dict["triangle"]
    node_coords = mesh.points
    numElements = len(S2)

    circ_true = np.empty(S2.shape, dtype=dctkit.float_dtype)
    circ_true[0, :] = np.array([0.5, 0, 0], dtype=dctkit.float_dtype)
    circ_true[1, :] = np.array([0, 0.5, 0], dtype=dctkit.float_dtype)
    circ_true[2, :] = np.array([1, 0.5, 0], dtype=dctkit.float_dtype)
    circ_true[3, :] = np.array([0.5, 1, 0], dtype=dctkit.float_dtype)
    for i in range(numElements):
        # FIXME: we are not checking the barycentric coordinates
        triangle_circ = circ.circumcenter(S2[i, :], node_coords)[0]
        assert triangle_circ.dtype == dctkit.float_dtype
        assert np.allclose(triangle_circ, circ_true[i, :])
