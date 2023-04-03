import numpy as np
import jax.numpy as jnp
from dctkit.dec import cochain as C
from dctkit.mesh import simplex as sim
import dctkit as dt
from dctkit.mesh import util


class ElasticaProblem():
    def __init__(self, num_elements: float, L: float, rho: float) -> None:
        self.num_elements = num_elements
        self.L = L
        self.rho = rho
        self.get_elastica_mesh()
        self.get_radius()

    def get_elastica_mesh(self) -> None:
        # load simplicial complex
        num_nodes = self.num_elements + 1
        S_1, x = util.generate_1_D_mesh(num_nodes, self.L)
        self.S = sim.SimplicialComplex(S_1, x, is_well_centered=True)
        self.S.get_circumcenters()
        self.S.get_primal_volumes()
        self.S.get_dual_volumes()
        self.S.get_hodge_star()

    def get_radius(self) -> None:
        r_node = (1 - (1 - self.rho)*self.S.node_coord[:, 0]/self.L)**4
        self.r = C.CochainP0(complex=self.S, coeffs=r_node)

    def energy_elastica(self, theta: np.array, EI0: np.array, theta_0: float, F: float) -> float:
        # add boundary conditions
        theta = jnp.insert(theta, 0, theta_0)
        # define A
        A = F*self.L**2/EI0
        # penalty = 0.5*gamma*(theta[0] - theta_true[0])**2
        # jax.debug.print("{E}", E=E[0])
        # define B
        # B = C.CochainD0(complex=S, coeffs=E*C.star(C.coboundary(I_coch)).coeffs)
        # f = E[0]*I_0
        internal_vec = np.ones(self.S.num_nodes, dtype=dt.float_dtype)
        internal_vec[0] = 0
        internal_vec[-1] = 0
        internal_coch = C.CochainP0(complex=self.S, coeffs=internal_vec)
        B_vec = self.r.coeffs
        B = C.CochainP0(complex=self.S, coeffs=B_vec)
        B_in = C.cochain_mul(B, internal_coch)
        # get dimensionless B
        theta = C.CochainD0(complex=self.S, coeffs=theta)
        const = C.CochainD0(complex=self.S, coeffs=A *
                            np.ones(self.S.num_nodes-1, dtype=dt.float_dtype))
        curvature = C.star(C.coboundary(theta))
        momentum = C.cochain_mul(B_in, curvature)
        energy = 0.5*C.inner_product(momentum, curvature) - \
            C.inner_product(const, C.sin(theta))
        return energy

    def obj_fun_theta(self, theta_guess: np.array, E_guess: np.array, theta_true: np.array) -> float:
        theta_guess = jnp.insert(theta_guess, 0, theta_true[0])
        # theta_guess = theta_guess[::density]
        obj = jnp.sum(jnp.square(theta_guess-theta_true))
        # jax.debug.print("{obj}", obj=obj)
        return obj
