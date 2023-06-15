import numpy.typing as npt
from dctkit.mesh.simplex import SimplicialComplex
import dctkit.dec.cochain as C
import dctkit.dec.vector as V
from jax import Array
import jax.numpy as jnp
from typing import Tuple
import jax


class LinearElasticity():
    """Linear elasticity class.

    Args:
        S: reference configuration simplicial complex.
        mu_: Lamé modulus
        lambda_: Lamé modulus
    """

    def __init__(self, S: SimplicialComplex, mu_: float, lambda_: float):
        self.S = S
        self.mu_ = mu_
        self.lambda_ = lambda_

    def get_strain(self, node_coords:  npt.NDArray | Array):
        current_metric = self.S.get_current_metric_2D(node_coords=node_coords)
        # define the infinitesimal strain and its trace
        epsilon = 1/2 * (current_metric - self.S.metric)
        return epsilon

    def get_stress(self, epsilon: npt.NDArray | Array):
        num_faces = self.S.S[2].shape[0]
        tr_epsilon = jnp.trace(epsilon, axis1=1, axis2=2)
        # get the stress via the consistutive equation for isotropic linear
        # elastic materials
        stress = 2*self.mu_*epsilon + self.lambda_*tr_epsilon[:, None, None] * \
            jnp.stack([jnp.identity(2)]*num_faces)
        return stress

    def linear_elasticity_residual(self, node_coords: C.CochainP0,
                                   f: C.CochainP2) -> C.CochainP2:
        """Compute the residual of the discrete balance equation in the case
          of isotropic linear elastic materials in 2D using DEC framework.

        Args:
            node_coords: primal vector valued 0-cochain of
            node coordinates of the current configuration.
            f: primal vector-valued 2-cochain of sources.

        Returns:
            the residual vector-valued cochain.

        """
        epsilon = self.get_strain(node_coords=node_coords.coeffs)
        stress = self.get_stress(epsilon=epsilon)
        stress_tensor = V.DiscreteTensorFieldD(S=self.S, coeffs=stress.T, rank=2)
        stress_integrated = V.flat_DPD(stress_tensor)
        # residual = C.add(C.coboundary(C.star(stress_integrated)), f)
        forces = C.star(stress_integrated)
        # force on free boundary is 0
        # FIXME: temporary setting. Free edges indexes for the square mesh:
        # [0,3,6,10]
        forces.coeffs = forces.coeffs.at[0].set(0)
        forces.coeffs = forces.coeffs.at[3].set(0)
        forces.coeffs = forces.coeffs.at[6].set(0)
        forces.coeffs = forces.coeffs.at[10].set(0)
        residual = C.add(C.coboundary(forces), f)
        return residual

    def obj_linear_elasticity(self, node_coords: npt.NDArray | Array,
                              f: npt.NDArray | Array, gamma: float, boundary_values:
                              Tuple[npt.NDArray, npt.NDArray]) -> float:
        """Objective function of the optimization problem associated to linear
           elasticity balance equation with Dirichlet boundary conditions on a portion
           of the boundary.

        Args:
            node_coords: matrix with node coordinates arranged row-wise.
            f: vector of external sources (constant term of the system).
            gamma: penalty factor.
            boundary_values: tuple of two np.arrays in which the first
            encodes the indices of boundary values, while the last encodes the
            boundary values themselves.

        Returns:
            the value of the objective function at node_coords.

        """
        node_coords_reshaped = node_coords.reshape(self.S.node_coord.shape)
        f = f.reshape((self.S.S[2].shape[0], self.S.embedded_dim-1))
        idx, value = boundary_values
        node_coords_coch = C.CochainP0(complex=self.S, coeffs=node_coords_reshaped)
        f_coch = C.CochainP2(complex=self.S, coeffs=f)
        residual = self.linear_elasticity_residual(node_coords_coch, f_coch).coeffs
        # penalty = jnp.sum((node_coords_reshaped[idx, :] - value)**2)
        # FIXME: temporary setting
        bnd_node_coords = node_coords_reshaped[idx, :]
        penalty = jnp.sum((bnd_node_coords[:, 0] - value[:, 0])**2) + \
            jnp.sum((bnd_node_coords[:, 1] - value[:, 1])**2)
        energy = 1/2*(jnp.sum(residual**2) + gamma*penalty)
        return energy
