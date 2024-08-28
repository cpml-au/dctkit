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

    circs_true = np.array(
        [[0.5, 0, 0], [0, 0.5, 0], [1, 0.5, 0], [0.5, 1, 0]], dtype=dctkit.float_dtype
    )
    bary_true = np.array(
        [[0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]],
        dtype=dctkit.float_dtype,
    )
    circs, bary = circ.circumcenter(S2, node_coords)
    assert circs.dtype == dctkit.float_dtype
    assert np.allclose(circs, circs_true)
    assert np.allclose(bary, bary_true)

    mesh, _ = util.generate_line_mesh(3)
    S2 = mesh.cells_dict["line"]
    node_coords = mesh.points
    circs, bary = circ.circumcenter(S2, node_coords)
    circs_true = np.array([[0.25, 0, 0], [0.75, 0.0, 0]], dtype=dctkit.float_dtype)
    bary_true = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert circs.dtype == dctkit.float_dtype
    assert np.allclose(circs, circs_true)
    assert np.allclose(bary, bary_true)
