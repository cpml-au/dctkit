import numpy as np
import dctkit as dt
import jax
from jax import jit, grad
from dctkit.dec import cochain as C
from dctkit.mesh import simplex, util
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
from dctkit.math.opt import optctrl
from scipy.optimize import minimize
import matplotlib.pyplot as plt

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


def test_elastica(is_bilevel=False):
    np.random.seed(42)

    num_nodes = 20
    S_1, x = util.generate_1_D_mesh(num_nodes)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    plt.plot(x, np.zeros(num_nodes))
    plt.show()

    B = 1.
    A = 2.
    gamma = 10000.
    theta_0 = 0.1*np.random.rand(num_nodes).astype(dt.float_dtype)

    def energy_elastica(theta: np.array, B: float) -> float:
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
                     args=(B), method="BFGS", jac=jac, options={'disp': 1}).x

    if is_bilevel:
        theta_true = theta

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            return jnp.sum(jnp.square(theta_guess-theta_true))
        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica, state_dim=num_nodes)
        B_0 = 0.2*np.ones(1, dtype=dt.float_dtype)
        theta, B, fval = prb.run(theta_0, B_0, tol=1e-2)
        assert fval < 1e-3

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
    test_elastica(is_bilevel=True)
