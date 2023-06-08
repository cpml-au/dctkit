import numpy as np
import numpy.typing as npt
from dctkit.mesh.simplex import SimplicialComplex
import dctkit.dec.cochain as C
import dctkit.dec.vector as V
from jax import Array
import jax.numpy as jnp
from typing import Tuple


class LinearElasticity():
    """Linear elasticity class.

    Args:
        S (SimplicialComplex): reference configuration simplicial complex.
        mu_ (float): Lamé modulus
        lambda_ (float): Lamé modulus 
    """

    def __init__(self, S: SimplicialComplex, mu_: float, lambda_: float):
        self.S = S
        self.mu_ = mu_
        self.lambda_ = lambda_

    def linear_elasticity_residual(self, node_coords: C.CochainP0,
                                   f: C.CochainP2) -> C.CochainP2:
        """Compute the residual of the discrete balance equation in the case
          of isotropic linear elastic materials in 2D using DEC framework.

          Args:
            node_coords (C.CochainP0): primal vector valued 0-cochain of
            node coordinates of the current configuration.
            f (C.CochainP2): primal vector-valued 2-cochain of sources.

          Returns:
            (C.CochainP2): residual vector-valued cochain.

        """
        dim = self.S.dim
        g = self.S.get_current_metric_2D(node_coords=node_coords.coeffs)
        g_tensor = V.DiscreteTensorFieldD(S=self.S, coeffs=g)
        # define the infinitesimal strain and its trace
        epsilon = 1/2 * (g_tensor - self.S.metric)
        tr_epsilon = jnp.trace(epsilon, axis1=1, axis2=2)
        # get the stress via the consistutive equation for isotropic linear
        # elastic materials
        stress = 2*self.mu*epsilon + self.lambda_*tr_epsilon[:, None, None] * \
            np.stack([np.identity(2)]*dim)
        stress_integrated = V.flat_DPD(stress)
        residual = C.coboundary(C.star(stress_integrated)) + f
        return residual

    def obj_linear_elasticity(self, node_coords: npt.NDArray | Array, f: npt.NDArray | Array,
                              gamma: float, boundary_values:
                              Tuple[npt.NDArray, npt.NDArray]) -> float:
        """Objective function of the optimization problem associated to linear elasticity
           balance equation with Dirichlet B.C. on a portion of the boundary.

           Args:
            node_coords (np.array): matrix with node coordinates arranged row-wise.
            f (np.array): vector of external sources (constant term of the system).
            gamma (float): penalty factor.
            boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the indices of boundary values, while the last encodes the
            boundary values themselves.

           Returns:
            (float): the value of the objective function at node_coords. 

        """
        idx, value = boundary_values
        node_coords_coch = C.CochainP0(S=self.S, coeffs=node_coords)
        f_coch = C.CochainP2(complex=self.S, coeffs=f)
        residual = self.linear_elasticity_residual(node_coords_coch, f_coch).coeffs
        penalty = jnp.sum((node_coords[idx] - value)**2)
        energy = 1/2*(jnp.linalg.norm(residual) + gamma*penalty)
        return energy
