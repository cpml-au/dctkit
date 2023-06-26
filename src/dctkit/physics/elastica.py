import numpy as np
import jax.numpy as jnp
from dctkit.dec import cochain as C
import dctkit as dt
from dctkit.mesh import util
import numpy.typing as npt
from jax import Array


class Elastica():
    """Euler's Elastica class.

    Args:
        num_elements: number of elements of primal mesh.
        L: length of the rod.
    """

    def __init__(self, num_elements: float, L: float) -> None:
        self.num_elements = num_elements
        self.L = L
        self.get_elastica_mesh()

    def get_elastica_mesh(self):
        """Constructs the normalized simplicial complex in the interval [0,1]."""

        num_nodes = self.num_elements + 1
        mesh, _ = util.generate_line_mesh(num_nodes, 1.)
        self.S = util.build_complex_from_mesh(mesh)
        self.S.get_hodge_star()

        # define internal cochain to compute the energy only in the interior points
        int_vec = np.ones(self.S.num_nodes, dtype=dt.float_dtype)
        int_vec[0] = 0
        int_vec[-1] = 0
        self.int_coch = C.CochainP0(complex=self.S, coeffs=int_vec)
        # define the unit cochain on the dual nodes
        self.ones_coch = C.CochainD0(complex=self.S,
                                     coeffs=np.ones(self.num_elements,
                                                    dtype=dt.float_dtype))

    def energy(self, theta: npt.NDArray, B: float, theta_0: float, F: float) -> Array:
        """Computes the total potential energy.

        Args:
            theta: current configuration angles.
            B: bending stiffness.
            theta_0: prescribed value of the angle at the left end (cantilever).
            F: vertical component of the applied force at the right end.

        Returns:
            value of the energy.

        """
        # apply bc at left end
        theta = jnp.insert(theta, 0, theta_0)
        theta_coch = C.CochainD0(complex=self.S, coeffs=theta)

        # define dimensionless load
        A = F*self.L**2/B

        # curvature at internal nodes
        curvature = C.cochain_mul(self.int_coch, C.star(C.coboundary(theta_coch)))

        # bending moment
        moment = C.scalar_mul(curvature, B)

        # potential of the applied load
        A_coch = C.scalar_mul(self.ones_coch, A)
        load = C.inner_product(C.sin(theta_coch), A_coch)

        energy = 0.5*C.inner_product(moment, curvature) - load

        return energy

    def obj_stiffness(self, theta: npt.NDArray, B: float,
                      theta_true: npt.NDArray) -> Array:
        """Objective function for the bending stiffness identification problem
            (inverse problem).

        Args:
            theta: candidate solution (except left angle).
            B: candidate bending stiffness.
            theta_true: true solution.

        Returns:
            error between the candidate and the true solutions.

        """
        theta = jnp.insert(theta, 0, theta_true[0])
        return jnp.sum(jnp.square(theta-theta_true))
