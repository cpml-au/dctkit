import numpy as np
import dctkit as dt
import jax
import jax.numpy as jnp
from jax import jit, grad
from dctkit.dec import cochain as C
from dctkit.mesh import simplex
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dctkit import config, FloatDtype, IntDtype, Backend, Platform

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


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

    plt.plot(x, np.zeros(num_nodes))
    plt.show()

    B = 1.
    A = 0.
    gamma = 10000.
    theta_0 = 0.1*np.random.rand(num_nodes).astype(dt.float_dtype)

    def energy_elastica(theta: np.array, A: float, B: float, gamma: float) -> float:
        theta = C.CochainD1(complex=S, coeffs=theta)
        const = C.CochainD1(complex=S, coeffs=A *
                            np.ones(num_nodes, dtype=dt.float_dtype))
        curvature = C.codifferential(theta)
        momentum = C.scalar_mul(curvature, B)
        energy = 0.5*C.inner_product(momentum, curvature) + \
            C.inner_product(const, C.sin(theta))
        penalty = 0.5*gamma*(theta.coeffs[0])**2
        final_energy = energy + penalty
        return final_energy

    obj = energy_elastica
    jac = jit(grad(obj))
    # get theta minimizing
    theta = minimize(fun=obj, x0=theta_0,
                     args=(A, B, gamma), method="BFGS", jac=jac, options={'disp': 1}).x
    # recover x and y
    x = np.empty(num_nodes)
    y = np.empty(num_nodes)
    x[0] = 0
    y[0] = 0
    h = 1/num_nodes
    for i in range(num_nodes-1):
        x[i + 1] = x[i] + h * np.cos(theta[i])
        y[i + 1] = y[i] + h * np.sin(theta[i])
    # plot the result
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    test_elastica()
