import numpy.typing as npt
from dctkit.mesh.simplex import SimplicialComplex
import dctkit.dec.cochain as C
import dctkit.dec.vector as V
from jax import Array
import jax.numpy as jnp
from typing import Tuple, Dict


class LinearElasticity():
    """Linear elasticity class.

    Args:
        S: reference configuration simplicial complex.
        mu_: Lamé modulus.
        lambda_: Lamé modulus.
    """

    def __init__(self, S: SimplicialComplex, mu_: float, lambda_: float):
        self.S = S
        self.mu_ = mu_
        self.lambda_ = lambda_

    def get_GreenLagrange_strain(self, node_coords:
                                 npt.NDArray | Array) -> npt.NDArray | Array:
        """Compute the discrete strain tensor given the current node coordinates.

        Args:
            node_coords: current node coordinates.

        Returns:
            the discrete strain tensor.

        """
        current_metric = self.S.get_current_metric_2D(node_coords=node_coords)
        # define the infinitesimal strain and its trace
        epsilon = 1/2 * (current_metric - self.S.reference_metric)
        return epsilon

    def get_stress(self, strain: npt.NDArray | Array) -> Array:
        """Compute the discrete stress tensor from strains using the consistutive
        equation for isotropic linear elastic materials.

        Args:
            strain: discrete strain tensor.

        Returns:
            the discrete stress tensor.

        """
        num_faces = self.S.S[2].shape[0]
        tr_strain = jnp.trace(strain, axis1=1, axis2=2)
        # get the stress via the consistutive equation for isotropic linear
        # elastic materials
        stress = 2.*self.mu_*strain + self.lambda_*tr_strain[:, None, None] * \
            jnp.stack([jnp.identity(2)]*num_faces)
        return stress

    def set_boundary_tractions(self, forces: C.CochainP1,
                               boundary_tractions:
                               Dict[str, Tuple[Array, Array]]) -> C.CochainP1:
        """Set the boundary tractions on primal edges.

        Args:
            forces: vector-valued primal 1-cochain containing forces acting on primal
                edges.
            boundary_tractions: a dictionary of tuples. Each key represent the type
                of coordinate to manipulate (x,y, or both), while each tuple consists
                of two jax arrays, in which the first encordes the indices where we
                want to impose the boundary tractions, while the last encodes the
                boundary traction values themselves.

        Returns:
            the updated force 1-cochain.

        """
        for key in boundary_tractions:
            idx, values = boundary_tractions[key]
            if key == ":":
                forces.coeffs = forces.coeffs.at[idx, :].set(values)
            else:
                forces.coeffs = forces.coeffs.at[idx, int(key)].set(values)
        return forces

    def force_balance_residual_primal(self, node_coords: C.CochainP0, f: C.CochainP2,
                                      boundary_tractions:
                                      Dict[str, Tuple[Array, Array]]) -> C.CochainP2:
        """Compute the residual of the discrete balance equation in the case
          of isotropic linear elastic materials in 2D using DEC framework.

        Args:
            node_coords: primal vector valued 0-cochain of
            node coordinates of the current configuration.
            f: primal vector-valued 2-cochain of sources.
            boundary_tractions: a dictionary of tuples. Each key represent the type
                of coordinate to manipulate (x,y, or both), while each tuple consists
                of two jax arrays, in which the first encordes the indices where we
                want to impose the boundary tractions, while the last encodes the
                boundary traction values themselves.

        Returns:
            the residual vector-valued cochain.

        """
        strain = self.get_GreenLagrange_strain(node_coords=node_coords.coeffs)
        stress = self.get_stress(strain=strain)
        stress_tensor = V.DiscreteTensorFieldD(S=self.S, coeffs=stress.T, rank=2)
        stress_integrated = V.flat_DPD(stress_tensor)
        forces = C.star(stress_integrated)
        # set tractions on given sub-portions of the boundary
        forces_bnd_update = self.set_boundary_tractions(forces, boundary_tractions)
        residual = C.add(C.coboundary(forces_bnd_update), f)
        return residual

    def force_balance_residual_dual(self, node_coords: C.CochainP0,
                                    f: C.CochainD2) -> C.CochainD2:
        """Compute the residual of the discrete balance equation in the case
          of isotropic linear elastic materials in 2D using DEC framework.

        Args:
            node_coords: primal vector valued 0-cochain of
            node coordinates of the current configuration.
            f: dual vector-valued 2-cochain of sources.

        Returns:
            the residual vector-valued cochain.

        """
        strain = self.get_GreenLagrange_strain(node_coords=node_coords.coeffs)
        stress = self.get_stress(strain=strain)
        stress_tensor = V.DiscreteTensorFieldD(S=self.S, coeffs=stress.T, rank=2)
        stress_integrated = V.flat_DPP(stress_tensor)
        forces = C.star(stress_integrated)
        print(forces.coeffs)
        print(C.coboundary(forces).coeffs)
        residual = C.add(C.coboundary(forces), f)
        return residual

    def obj_linear_elasticity(self, node_coords: npt.NDArray | Array,
                              f: npt.NDArray | Array, gamma: float, boundary_values:
                              Dict[str, Tuple[Array, Array]],
                              boundary_tractions: Dict[str, Tuple[Array,
                                                                  Array]]) -> float:
        """Objective function of the optimization problem associated to linear
           elasticity balance equation with Dirichlet boundary conditions on a portion
           of the boundary.

        Args:
            node_coords: 1-dimensional array obtained after flattening
            the matrix with node coordinates arranged row-wise.
            f: 1-dimensional array obtained after flattening the
            matrix of external sources (constant term of the system).
            gamma: penalty factor.
            boundary_values: a dictionary of tuples. Each key represent the type of
                coordinate to manipulate (x,y, or both), while each tuple consists of
                two np.arrays in which the first encodes the indices of boundary
                values, while the last encodes the boundary values themselves.
            boundary_tractions: a dictionary of tuples. Each key represent the type
                of coordinate to manipulate (x,y, or both), while each tuple consists
                of two jax arrays, in which the first encordes the indices where we want
                to impose the boundary tractions, while the last encodes the boundary
                traction values themselves. It is None when we perform the force balance
                on dual cells.

        Returns:
            the value of the objective function at node_coords.

        """
        node_coords_reshaped = node_coords.reshape(self.S.node_coords.shape)
        node_coords_coch = C.CochainP0(complex=self.S, coeffs=node_coords_reshaped)
        if boundary_tractions is None:
            f = f.reshape((self.S.num_nodes, self.S.space_dim-1))
            f_coch = C.CochainD2(complex=self.S, coeffs=f)
            residual = self.force_balance_residual_dual(node_coords_coch, f_coch).coeffs
        else:
            f = f.reshape((self.S.S[2].shape[0], self.S.space_dim-1))
            f_coch = C.CochainP2(complex=self.S, coeffs=f)
            residual = self.force_balance_residual_primal(
                node_coords_coch, f_coch, boundary_tractions).coeffs
        penalty = 0.
        for key in boundary_values:
            idx, values = boundary_values[key]
            if key == ":":
                penalty += jnp.sum((node_coords_reshaped[idx, :] - values)**2)
            else:
                penalty += jnp.sum((node_coords_reshaped[idx, :]
                                   [:, int(key)] - values)**2)
        energy = 1/2*(jnp.sum(residual**2) + gamma*penalty)
        return energy
