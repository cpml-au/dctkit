import numpy as np
import dctkit as dt
import jax
import jax.numpy as jnp
from jax import jit, grad
from dctkit.dec import cochain as C
from dctkit.mesh import simplex, util
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
from dctkit.math.opt import optctrl
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


def test_elastica(is_bilevel=False):
    np.random.seed(42)

    # load FEM solution for benchmark
    filename = os.path.join(os.path.dirname(__file__), "theta_bench_FEM.txt")
    theta_exact = np.genfromtxt(filename)

    density = 1
    num_fem_nodes = 100
    num_nodes = density*num_fem_nodes + 1
    print(num_nodes)
    S_1, x = util.generate_1_D_mesh(num_nodes)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    #plt.plot(x, np.zeros(num_nodes))
    # plt.show()

    B = 1.
    A = -4.
    gamma = 100000000.
    theta_0 = 0.1*np.random.rand(num_nodes).astype(dt.float_dtype)
    #theta_0[0] = 0

    def energy_elastica(theta: np.array, B: float) -> float:
        theta = C.CochainD1(complex=S, coeffs=theta)
        const = C.CochainD1(complex=S, coeffs=A *
                            np.ones(num_nodes, dtype=dt.float_dtype))
        curvature = C.codifferential(theta)
        momentum = C.scalar_mul(curvature, B)
        energy = 0.5*C.inner_product(momentum, curvature) - \
            C.inner_product(const, C.sin(theta))
        #penalty = 0.5*gamma*(theta.coeffs[0])**2
        #final_energy = energy + penalty
        final_energy = energy
        # print(final_energy)
        return final_energy

    obj = energy_elastica
    jac = jit(grad(obj))

    # define linear constraint theta(0) = 0
    #constr = [0]*num_nodes
    #constr[0] = 1
    #lin_constr = LinearConstraint(constr, lb=[0], ub=[0])
    cons = ({'type': 'eq', 'fun': lambda x: x[0]})

    if is_bilevel:
        theta_true = theta_exact[:, 1]

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            return jnp.sum(jnp.square(theta_guess-theta_true))
        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica, state_dim=num_nodes)
        B_0 = 0.2*np.ones(1, dtype=dt.float_dtype)
        theta, B, fval = prb.run(theta_0, B_0, tol=1e-2)
        print(fval)
        #assert fval < 1e-3

    # get theta minimizing
    if not is_bilevel:
        # theta = minimize(fun=obj, x0=theta_0,
        # args=(B), method="trust-constr", jac=jac, constraints=lin_constr, options={'disp': 1}).x
        theta = minimize(fun=obj, x0=theta_0,
                         args=(B), method="SLSQP", jac=jac, constraints=cons, options={'disp': 1, 'maxiter': 500}).x
        theta = theta[::density]
        print(theta[-1])
        print(theta_exact[-1, 1])
        #print((theta[-1] - theta_exact[-1, 1])/theta_exact[-1, 1])
        print(np.linalg.norm(theta - theta_exact[:, 1]))

    # recover x_true and y_true
    x_true = np.empty(num_fem_nodes + 1)
    y_true = np.empty(num_fem_nodes + 1)
    x_true[0] = 0
    y_true[0] = 0
    h = 1/num_fem_nodes
    for i in range(num_fem_nodes):
        x_true[i + 1] = x_true[i] + h * np.cos(theta_exact[i, 1])
        y_true[i + 1] = y_true[i] + h * np.sin(theta_exact[i, 1])

    # recover x and y
    x = np.empty(num_fem_nodes + 1)
    y = np.empty(num_fem_nodes + 1)
    x[0] = 0
    y[0] = 0
    h = 1/num_fem_nodes
    for i in range(num_fem_nodes):
        x[i + 1] = x[i] + h * np.cos(theta[i])
        y[i + 1] = y[i] + h * np.sin(theta[i])

    # plot the results
    plt.plot(x_true, y_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()

    plt.plot(S.node_coord[:, 0], theta_exact[:, 1], 'r')
    plt.plot(S.node_coord[:, 0], theta, 'b')
    plt.show()


if __name__ == "__main__":
    test_elastica(is_bilevel=False)
