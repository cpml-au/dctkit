import numpy as np
import dctkit
import numpy.typing as npt
from typing import Tuple

# FIXME: pass only the coordinates of the nodes of the simplex or consider making this a
# method of the SimplicialComplex class.


def circumcenter(S: npt.NDArray,
                 node_coords: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute the circumcenter of a set of simplices (arranged row-wise).
        (Reference: Bell, Hirani, PyDEC: Software and Algorithms for Discretization
        of Exterior Calculus, 2012, Section 10.1).

        Args:
            S: matrix containing the IDs of the nodes belonging to each simplex.
            node_coords: coordinates (cols) of each node of the complex to which the
                simplices belongs.
        Returns:
            a tuple consisting of the Cartesian coordinates of the circumcenter and the
                barycentric coordinates.
    """
    # get global data type
    float_dtype = dctkit.float_dtype

    # store the coordinates of the nodes for each simplex in S
    S_coord = node_coords[S]

    transpose_S_coord = np.transpose(S_coord, [0, 2, 1])
    # extract number of p-simplices and number of nodes per simplex
    num_p, num_nodes_per_spx = S.shape

    # construct for each simplex s the matrix
    # A_s = [B_s | 1_{m-1,1};
    #        1_{1, m-1} |  0]
    # where m-1 is the number of nodes per simplex and B is the matrix (m-1)x(m-1)
    # defined as B_ij = 2 <v_i, v_j>
    A = np.zeros((num_p, num_nodes_per_spx+1, num_nodes_per_spx+1), dtype=float_dtype)
    A[:, :num_nodes_per_spx, :num_nodes_per_spx] = 2 * S_coord @ transpose_S_coord
    A[:, :, -1] = 1
    A[:, -1, :] = 1
    A[:, -1, -1] = 0

    # the vector of constant terms b for a given p-simplex s = {v_0,...,v_p} is
    # [<v_0,v_0>, ..., <v_p,v_p>, 1], where v_i is the coordinate of the
    # i-th node in s
    b = np.ones((num_p, num_nodes_per_spx+1), dtype=float_dtype)
    b[:, range(num_nodes_per_spx)] = np.sum(S_coord * S_coord, axis=2)

    # barycentric coordinates x of the circumcenter are the solution
    # of the linear sistem Ax = b without the last component (i.e. x[:-1])
    bary_coords_extended = np.linalg.solve(A, b)
    bary_coords = bary_coords_extended[:, :-1]

    # the circumcenter of a p-simplex {v0,...,vp} can be written in barycentric
    # coordinates as c = sum_j x_j v_j
    circumcenter = np.einsum("ij,ijk -> ik", bary_coords, S_coord)
    return circumcenter, bary_coords
