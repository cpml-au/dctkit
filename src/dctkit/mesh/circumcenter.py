import numpy as np
import dctkit
import numpy.typing as npt
from typing import Tuple

# FIXME: pass only the coordinates of the nodes of the simplex or consider making this a
# method of the SimplicialComplex class.


def circumcenter(s: npt.NDArray,
                 node_coords: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute the circumcenter of a given simplex. (Reference: Bell, Hirani,
        PyDEC: Software and Algorithms for Discretization of Exterior Calculus,
        2012, Section 10.1).

        Args:
            s: array containing the IDs of the nodes beloging to the given simplex.
            node_coords: coordinates (cols) of each node of the complex to which the
                given simplex belongs.
        Returns:
            a tuple consisting of the Cartesian coordinates of the circumcenter and the
                barycentric coordinates.
    """
    # get global data type
    float_dtype = dctkit.float_dtype

    # store the coordinates of the nodes in s
    simplex_coord = node_coords[s[:]]
    rows, cols = simplex_coord.shape

    assert (rows <= cols + 1)

    # construct the matrix A
    A = np.bmat([[2*np.dot(simplex_coord, simplex_coord.T),
                  np.ones((rows, 1), dtype=float_dtype)],
                [np.ones((1, rows), dtype=float_dtype),
                 np.zeros((1, 1), dtype=float_dtype)]])
    b = np.hstack((np.sum(simplex_coord * simplex_coord, axis=1),
                   np.ones((1), dtype=float_dtype)))

    # barycentric coordinates x of the circumcenter are the solution
    # of the linear sistem Ax = b
    bary_coords = np.linalg.solve(A, b)
    bary_coords = bary_coords[:-1]

    # compute the coordinates of the circumcenter
    circumcenter = np.dot(bary_coords, simplex_coord)

    return circumcenter, bary_coords
