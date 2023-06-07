import numpy as np
import numpy.typing as npt
from dctkit.mesh.simplex import SimplicialComplex
import dctkit.dec.cochain as C
import dctkit.dec.vector as V
from jax import Array
import jax.numpy as jnp
from typing import Tuple


class LinearElasticity():
    def __init__(self, S: SimplicialComplex, mu_: float, lambda_: float):
        self.S = S
        self.mu_ = mu_
        self.lambda_ = lambda_

    def __get_current_metric_2D(self, node_coords: npt.Array | Array):
        """Compute the multiarray of shape (n, 2, 2) where n is the number of
            2-simplices and any 2x2 matrix is the metric of the corresponding 2-simplex.
        """
        dim = self.S.dim
        B = self.S.B[dim]
        primal_edges = self.S.S[1]
        # construct the matrix in which the i-th row corresponds to the vector
        # of coordinates of the i-th primal edge
        primal_edge_vectors = node_coords[primal_edges[:, 1], :] - \
            node_coords[primal_edges[:, 0], :]
        # construct the multiarray of shape (n, 3, 3) where any 3x3 matrix represents
        # the coordinates of the edge vectors (arranged in rows) belonging to the
        # corresponding 2-simplex
        primal_edges_per_2_simplex = primal_edge_vectors[B]
        # extract the first two rows, i.e. basis vectors, for each 3x3 matrix
        basis_vectors = primal_edges_per_2_simplex[:, :-1, :]
        metric = basis_vectors @ np.transpose(
            basis_vectors, axes=(0, 2, 1))
        return metric

    def linear_elasticity_residual(self, node_coords: C.CochainP0,
                                   f: C.CochainP2) -> C.CochainP2:
        dim = self.S.dim
        g = self.__get_current_metric_2D(node_coords=node_coords.coeffs)
        g_tensor = V.DiscreteTensorFieldD(S=self.S, coeffs=g)
        epsilon = 1/2 * (g_tensor - self.S.metric)
        tr_epsilon = jnp.trace(epsilon, axis1=1, axis2=2)
        stress = 2*self.mu*epsilon + self.lambda_*tr_epsilon[:, None, None] * \
            np.stack([np.identity(2)]*dim)
        stress_integrated = V.flat_DPD(stress)
        residual = C.coboundary(C.star(stress_integrated)) + f
        return residual

    def obj_linear_elasticity(self, node_coords: npt.Array, f: npt.Array, gamma: float,
                              boundary_values: Tuple[npt.NDArray, npt.NDArray]):
        pos, value = boundary_values
        node_coords_coch = C.CochainP0(S=self.S, coeffs=node_coords)
        f_coch = C.CochainP2(complex=self.S, coeffs=f)
        residual = self.linear_elasticity_residual(node_coords_coch, f_coch).coeffs
        penalty = np.sum((node_coords[pos] - value)**2)
