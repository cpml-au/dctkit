import numpy as np


def unsigned_volume(S, node_coord):
    """Compute the unsigned volume of a set of simplices S.

    Args:
        S (np.array): matrix containing the IDs of the nodes belonging to each simplex.
        node_coord (np.array): coordinates of every node of the cell complex
                               in which s is defined.
    Returns:
        vol (float): unsigned volume of the simplex.
    """
    # store the coordinates of the nodes of every simplex in S
    S_coord = node_coord[S]
    rows = S_coord.shape[1]

    # indices to extract the matrix with rows equal to the rows of S with indices
    # non-congruent to 0 modulo rows-1
    index = 1 + np.array(range(rows-1))

    # compute the matrix of matrices V
    V = S_coord[:, index, :] - S_coord[:, ::(rows), :]

    # compute the transpose of V with respect to the last two axes
    transpose_V = np.transpose(V, [0, 2, 1])

    VTV = np.matmul(V, transpose_V)
    # math formula to compute the unsigned volume of a simplex
    vol = np.sqrt(np.abs(np.linalg.det(VTV)))/np.math.factorial(rows - 1)
    return vol


def signed_volume(S, node_coord):
    """Compute the signed volume of a set of n-simplices in an n-simplicial complex.

    Args:
        S (np.array): matrix containing the IDs of the nodes belonging to each simplex.
        node_coord (np.array): coordinates of every node of the cell complex
                               in which s is defined.
    Returns:
        vol (float): signed volume of the simplex.
    """
    S_coord = node_coord[S]
    _, rows, cols = S_coord.shape

    assert (rows == cols + 1)
    # indices to extract the matrix with rows equal to the rows of S with indices
    # non-congruent to 0 modulo rows-1
    index = 1 + np.array(range(rows-1))

    # compute the matrix of matrices V
    V = S_coord[:, index, :] - S_coord[:, ::(rows), :]

    # math formula to compute the signed volume of a simplex
    vol = np.linalg.det(V) / np.math.factorial(rows-1)
    return vol
