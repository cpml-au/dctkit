import numpy as np
import dctkit as dt
import jax
import jax.numpy as jnp
from jax import jit, grad, jacrev
from dctkit.dec import cochain as C
from dctkit.mesh import simplex, util
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
from dctkit.math.opt import optctrl
from scipy.optimize import minimize
import matplotlib.pyplot as plt

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


def test_elastica(is_bilevel=False):
    np.random.seed(42)

    # load FEM solution for benchmark
    filename = os.path.join(os.path.dirname(__file__), "theta_bench_FEM.txt")
    theta_exact = np.genfromtxt(filename)

    # load simplicial complex
    density = 1
    num_fem_elements = 100
    num_nodes = density*num_fem_elements + 1
    S_1, x = util.generate_1_D_mesh(num_nodes)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    # modify hodge star inverse at the boundary
    S.hodge_star_inverse[0][0] /= 2
    S.hodge_star_inverse[0][-1] /= 2

    # set params
    A = -4.
    theta_0 = 0.1*np.random.rand(num_nodes).astype(dt.float_dtype)

    def energy_elastica(theta: np.array, B: float) -> float:
        theta = C.CochainD1(complex=S, coeffs=theta)
        const = C.CochainD1(complex=S, coeffs=A *
                            np.ones(num_nodes, dtype=dt.float_dtype))
        curvature = C.codifferential(theta)
        momentum = C.scalar_mul(curvature, B)
        energy = 0.5*C.inner_product(momentum, curvature) - \
            C.inner_product(const, C.sin(theta))

        return energy

    obj = energy_elastica
    jac = jit(grad(obj))

    # define linear constraint theta(0) = 0, delta_theta[-1] = 0
    cons = ({'type': 'eq', 'fun': lambda x: x[0]},
            {'type': 'eq', 'fun': lambda x: x[num_nodes-2] - x[num_nodes-1]})

    if is_bilevel:
        theta_true = theta_exact[:, 1]

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            return jnp.sum(jnp.square(theta_guess-theta_true))
        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica, state_dim=num_nodes)
        B_0 = 0.1*np.ones(1, dtype=dt.float_dtype)
        theta, B, fval = prb.run(theta_0, B_0, tol=1e-7)
        print(B)
        print(fval)
        print(theta)
        #assert fval < 1e-3

    # get theta minimizing
    if not is_bilevel:
        B = 1.
        res = minimize(fun=obj, x0=theta_0, args=(B), method="SLSQP",
                       jac=jac, constraints=cons, options={'disp': 1, 'maxiter': 500})
        print(res)
        theta = res.x
        theta = theta[::density]
        print(np.linalg.norm(theta - theta_exact[:, 1]))

    # recover x_true and y_true
    x_true = np.empty(num_fem_elements + 1)
    y_true = np.empty(num_fem_elements + 1)
    x_true[0] = 0
    y_true[0] = 0
    h = 1/num_fem_elements
    for i in range(num_fem_elements):
        x_true[i + 1] = x_true[i] + h * np.cos(theta_exact[i, 1])
        y_true[i + 1] = y_true[i] + h * np.sin(theta_exact[i, 1])

    # recover x and y
    x = np.empty(num_fem_elements + 1)
    y = np.empty(num_fem_elements + 1)
    x[0] = 0
    y[0] = 0
    h = 1/num_fem_elements
    for i in range(num_fem_elements):
        x[i + 1] = x[i] + h * np.cos(theta[i])
        y[i + 1] = y[i] + h * np.sin(theta[i])

    # plot the results
    plt.plot(x_true, y_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()

    node_coord = S.node_coord[:, 0]
    node_coord = node_coord[::density]
    plt.plot(node_coord, theta_exact[:, 1], 'r')
    plt.plot(node_coord, theta, 'b')
    plt.show()


if __name__ == "__main__":
    test_elastica(is_bilevel=False)
