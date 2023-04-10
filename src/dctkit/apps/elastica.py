import numpy as np
import jax.numpy as jnp
from dctkit.dec import cochain as C
from dctkit.mesh import simplex as sim
import dctkit as dt
from dctkit.mesh import util
import numpy.typing as npt
from jax import Array


class ElasticaProblem():
    """Elastica problem class.

    Args:
        num_elements (float): number of elements of primal mesh.
        L (float): length of the rod.
        rho (float): ratio between final and initial radius of the section of the rod.
    """

    def __init__(self, num_elements: float, L: float, rho: float) -> None:
        self.num_elements = num_elements
        self.L = L
        self.rho = rho
        self.get_elastica_mesh()
        self.get_radius()

    def get_elastica_mesh(self) -> None:
        """Routine to construct the normalized simplicial complex"""
        # load simplicial complex
        num_nodes = self.num_elements + 1
        S_1, x = util.generate_1_D_mesh(num_nodes, 1)
        self.S = sim.SimplicialComplex(S_1, x, is_well_centered=True)
        self.S.get_circumcenters()
        self.S.get_primal_volumes()
        self.S.get_dual_volumes()
        self.S.get_hodge_star()

    def get_radius(self) -> None:
        """Routine to compute the radius vector"""
        r_node = (1 - (1 - self.rho)*self.S.node_coord[:, 0])**4
        self.r = C.CochainP0(complex=self.S, coeffs=r_node)

    def energy_elastica(self, theta: npt.NDArray, EI0: npt.NDArray, theta_0: float,
                        F: float) -> float:
        """Routine that compute the elastica energy.

        Args:
            theta (np.array): current configuration angles.
            EI0 (np.array): product between E and I_0.
            theta_0 (float): value of theta in the first primal node (boundary
            condition).
            F (float): value of the force.

        Returns:
            float: the value of the energy.

        """
        # add boundary conditions
        theta = jnp.insert(theta, 0, theta_0)
        # define A
        A = F*self.L**2/EI0
        # define internal cochain to compute the energy only in the interior points
        internal_vec = np.ones(self.S.num_nodes, dtype=dt.float_dtype)
        internal_vec[0] = 0
        internal_vec[-1] = 0
        internal_coch = C.CochainP0(complex=self.S, coeffs=internal_vec)
        # define B
        B_vec = self.r.coeffs
        B = C.CochainP0(complex=self.S, coeffs=B_vec)
        B_in = C.cochain_mul(B, internal_coch)
        theta_coch = C.CochainD0(complex=self.S, coeffs=theta)
        const = C.CochainD0(complex=self.S, coeffs=A *
                            np.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
        # get curvature and momementum
        curvature = C.star(C.coboundary(theta_coch))
        momentum = C.cochain_mul(B_in, curvature)
        energy = 0.5*C.inner_product(momentum, curvature) - \
            C.inner_product(const, C.sin(theta_coch))
        return energy

    def obj_fun_theta(self, theta_guess: npt.NDArray, EI_guess: npt.NDArray,
                      theta_true: npt.NDArray) -> Array:
        """Objective function for the bilevel problem (inverse problem).

        Args:
            theta_guess: candidate solution.
            EI_guess: candidate EI.
            theta_true: true solution.

        Returns:
            float: error between the candidate and the true theta

        """
        theta_guess = jnp.insert(theta_guess, 0, theta_true[0])
        return jnp.sum(jnp.square(theta_guess-theta_true))
