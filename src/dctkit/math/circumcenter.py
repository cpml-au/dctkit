import numpy as np
import dctkit


def circumcenter(s, node_coord):
    """Compute the circumcenter of a given simplex s. (References: Bell, Hirani,
        PyDEC: Software and Algorithms for Discretization of Exterior Calculus,
        2012, Section 10.1).

        Args:
            s (np.array): array containing the IDs of the nodes beloging to s.
            node_coord (np.array): coordinates of every node of the cell complex
                in which s is defined.
        Returns:
            np.array: coordinates of the circumcenter of s.
    """
    # get global data type
    float_dtype = dctkit.float_dtype

    # store the coordinates of the nodes in s
    simplex_coord = node_coord[s[:]]
    rows, cols = simplex_coord.shape

    assert (rows <= cols + 1)

    # construct the matrix A
    A = np.bmat([[2*np.dot(simplex_coord, simplex_coord.T), np.ones((rows, 1), dtype=float_dtype)],
                [np.ones((1, rows), dtype=float_dtype), np.zeros((1, 1), dtype=float_dtype)]])
    b = np.hstack((np.sum(simplex_coord * simplex_coord, axis=1),
                   np.ones((1), dtype=float_dtype)))

    # barycentric coordinates x of the circumcenter are the solution
    # of the linear sistem Ax = b
    bary_coords = np.linalg.solve(A, b)
    bary_coords = bary_coords[:-1]

    # compute the coordinates of the circumcenter
    circumcenter = np.dot(bary_coords, simplex_coord)

    return circumcenter, bary_coords
