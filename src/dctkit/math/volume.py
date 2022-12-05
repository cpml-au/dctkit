import numpy as np


def unsigned_volume(s, node_coord):
    # store the coordinates of the nodes in s
    simplex_coord = node_coord[s[:]]
    rows, _ = simplex_coord.shape
    if rows == 1:
        return 1.0
    V = simplex_coord[1:, :] - simplex_coord[0, :]
    return np.sqrt(np.abs(np.linalg.det(np.dot(V, V.T)))) / np.math.factorial(rows-1)


def signed_volume(s, node_coord):
    simplex_coord = node_coord[s[:]]
    rows, cols = simplex_coord.shape
    assert (rows == cols + 1)
    V = simplex_coord[1:, :] - simplex_coord[0, :]
    return np.linalg.det(V) / np.math.factorial(rows-1)
