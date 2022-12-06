import numpy as np


def unsigned_volume(s, node_coord):
    """Compute the unsigned volume of a simplex.

    Args:
        s (np.array): array containing the IDs of the nodes beloging to s.
        node_coord (np.array): coordinates of every node of the cell complex
                               in which s is defined.
    Returns:
        vol (float): unsigned volume of the simplex.
    """
    # store the coordinates of the nodes in s
    simplex_coord = node_coord[s[:]]
    rows, _ = simplex_coord.shape
    if rows == 1:
        return 1.0
    V = simplex_coord[1:, :] - simplex_coord[0, :]
    # math formula to compute the unsigned volume of a simplex
    vol = np.sqrt(np.abs(np.linalg.det(np.dot(V, V.T)))) / np.math.factorial(rows-1)
    return vol


def signed_volume(s, node_coord):
    """Compute the signed volume of an n-simplex in an n-simplicial complex.

    Args:
        s (np.array): array containing the IDs of the nodes beloging to s.
        node_coord (np.array): coordinates of every node of the cell complex
                               in which s is defined.
    Returns:
        vol (float): signed volume of the simplex.
    """
    simplex_coord = node_coord[s[:]]
    rows, cols = simplex_coord.shape
    assert (rows == cols + 1)
    V = simplex_coord[1:, :] - simplex_coord[0, :]
    # math formula to compute the signed volume of a simplex
    vol = np.linalg.det(V) / np.math.factorial(rows-1)
    return vol
