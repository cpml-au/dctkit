import numpy as np
import dctkit as dt
import jax
from dctkit.dec import cochain as C
from dctkit.mesh import simplex


def test_elastica():
    if jax.config.read("jax_enable_x64"):
        assert dt.float_dtype == "float64"

    np.random.seed(42)

    num_nodes = 20
    node_coords = np.linspace(0, 1, num=num_nodes)
    x = np.zeros((num_nodes, 2))
    x[:, 0] = node_coords
    # define the Simplicial complex
    S_1 = np.empty((num_nodes - 1, 2))
    S_1[:, 0] = np.arange(num_nodes-1)
    S_1[:, 1] = np.arange(1, num_nodes)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    B = 1.
    A = 2.
    gamma = 1000.

    def energy_elastica(theta: np.array, A: float, B: float, gamma: float) -> float:
        theta = C.Cochain(dim=1, is_primal=False, complex=S, coeffs=theta)
        const = C.Cochain(dim=1, is_primal=False, complex=S,
                          coeffs=A*np.ones(num_nodes))
        curvature = C.codifferential(theta)
        momentum = C.scalar_mul(curvature, B)
        energy = 0.5*C.inner_product(momentum, curvature) + \
            C.inner_product(const, C.sin(theta))
        penalty = 0.5*gamma*dt.backend.sum((theta.coeffs[0])**2)
        final_energy = energy + penalty
        return final_energy
