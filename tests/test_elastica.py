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
import os

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


def get_elastica_mesh(density, num_fem_elements):
    # load simplicial complex
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
    return S


def test_elastica(is_bilevel=False):
    np.random.seed(42)

    # load FEM solution for benchmark
    filename = os.path.join(os.path.dirname(__file__), "theta_bench_FEM.txt")
    theta_exact = np.genfromtxt(filename)

    density = 1
    num_fem_elements = 100
    S = get_elastica_mesh(density, num_fem_elements)
    num_nodes = S.num_nodes

    # set params
    A = -4.

    if is_bilevel:
        theta_true = theta_exact[:, 1]
        theta_0 = 0.1*np.random.rand(num_nodes-2).astype(dt.float_dtype)

        def energy_elastica_constr(theta: np.array, B: float) -> float:
            theta = jnp.insert(theta, 0, 0)
            theta = jnp.append(theta, theta[-1])
            theta = C.CochainD1(complex=S, coeffs=theta)
            const = C.CochainD1(complex=S, coeffs=A *
                                np.ones(num_nodes, dtype=dt.float_dtype))
            curvature = C.codifferential(theta)
            momentum = C.scalar_mul(curvature, B)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            return energy

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            theta_guess = jnp.insert(theta_guess, 0, 0)
            theta_guess = jnp.append(theta_guess, theta_guess[-1])
            return jnp.sum(jnp.square(theta_guess-theta_true))
        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica_constr, state_dim=num_nodes-2)
        B_0 = 0.3*np.ones(1, dtype=dt.float_dtype)
        theta, B, fval = prb.run(theta_0, B_0, tol=1e-2)
        # extend theta
        theta = np.insert(theta, 0, 0)
        theta = np.append(theta, theta[-1])
        print(f"The optimal B is {B[0]}")
        print(fval)
        # assert fval < 1e-3

    else:
        B = 1.
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

        jac = jit(grad(energy_elastica))

        # define linear constraint theta(0) = 0, delta_theta[-1] = 0
        cons = ({'type': 'eq', 'fun': lambda x: x[0]},
                {'type': 'eq', 'fun': lambda x: x[num_nodes-2] - x[num_nodes-1]})

        res = minimize(fun=energy_elastica, x0=theta_0, args=(B), method="SLSQP",
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


def test_elastica_data(data_type, is_bilevel=False):
    np.random.seed(42)
    # load true data
    filename = os.path.join(os.path.dirname(__file__), "Data1_elastica.csv")
    data = np.genfromtxt(filename, delimiter=',')
    # submatrix of all x and y
    data_xy = data[2:, 1:]
    # submatrix of F
    data_F = data[0, :]
    data_F = data_F[1::2]
    x_true, y_true = data_xy[:, 2*data_type], data_xy[:, 2*data_type + 1]

    density = 1
    num_fem_elements = 10
    h = 1/num_fem_elements
    R_0 = 1.2
    R_1 = 0.3
    L = 1.524
    rho = R_1/R_0
    F = data_F[data_type]
    S = get_elastica_mesh(density, num_fem_elements)
    node_coord = S.node_coord[:, 0]
    num_nodes = S.num_nodes

    # define I_0
    I_0 = np.pi/4 * R_0**4
    # define I
    # I = I_0*(1 + 1/L*(1 - rho)*node_coord)**4
    I = I_0*(1 + 1/L*(1 - rho) * 1/num_fem_elements)**4 * \
        np.ones(num_fem_elements*density, dtype=dt.float_dtype)
    # I_coch = C.CochainP0(complex=S, coeffs=I)
    I_coch = C.CochainD0(complex=S, coeffs=I)

    if is_bilevel:
        # get theta_true
        theta_true = np.empty(num_nodes)
        theta_true[0] = 0
        for i in range(num_fem_elements-1):
            theta_true[i + 1] = np.arccos((x_true[i + 1] - x_true[i])/h)

        theta_true[-1] = theta_true[-2]

        print(theta_true)
        theta_0 = 0.1*np.random.rand(num_nodes-2).astype(dt.float_dtype)

        def energy_elastica_constr(theta: np.array, E: float) -> float:
            theta = jnp.insert(theta, 0, 0)
            theta = jnp.append(theta, theta[-1])
            # define A
            A = F*L**2/(E*I_0)
            # define B
            # B = C.CochainD0(complex=S, coeffs=E*C.star(C.coboundary(I_coch)).coeffs)
            B = C.CochainD0(complex=S, coeffs=E*I_coch.coeffs)
            theta = C.CochainD1(complex=S, coeffs=theta)
            const = C.CochainD1(complex=S, coeffs=A *
                                np.ones(num_nodes, dtype=dt.float_dtype))
            curvature = C.codifferential(theta)
            momentum = C.cochain_mul(B, curvature)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            return energy

        '''
        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            theta_guess = jnp.insert(theta_guess, 0, 0)
            theta_guess = jnp.append(theta_guess, theta_guess[-1])
            # recover x and y
            x = np.empty(num_fem_elements + 1)
            y = np.empty(num_fem_elements + 1)
            x[0] = 0
            y[0] = 0
            for i in range(num_fem_elements):
                x[i + 1] = x[i] + h * np.cos(theta_guess[i])
                y[i + 1] = y[i] + h * np.sin(theta_guess[i])
            return jnp.sum(jnp.square(x-x_true)) + jnp.sum(jnp.square(y-y_true))
        '''

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            theta_guess = jnp.insert(theta_guess, 0, 0)
            theta_guess = jnp.append(theta_guess, theta_guess[-1])
            return jnp.sum(jnp.square(theta_guess-theta_true))

        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica_constr, state_dim=num_nodes-2)
        E_0 = 6*np.ones(1, dtype=dt.float_dtype)
        theta, E, fval = prb.run(theta_0, E_0, tol=1e-2)
        # extend theta
        theta = np.insert(theta, 0, 0)
        theta = np.append(theta, theta[-1])
        print(f"The optimal E is {E[0]}")
        print(fval)
        # assert fval < 1e-3

    else:
        E_all = np.array([6., 5.6, 6.5, 5.8])
        E = E_all[data_type]
        theta_0 = 1*np.random.rand(num_nodes).astype(dt.float_dtype)

        def energy_elastica(theta: np.array, E: float) -> float:
            # define A
            A = F*L**2/(E*I_0)
            # define B
            # B = C.CochainD0(complex=S, coeffs=E*C.star(C.coboundary(I_coch)).coeffs)
            B = C.CochainD0(complex=S, coeffs=E*I_coch.coeffs)
            # print(B.coeffs)
            theta = C.CochainD1(complex=S, coeffs=theta)
            const = C.CochainD1(complex=S, coeffs=A *
                                np.ones(num_nodes, dtype=dt.float_dtype))
            curvature = C.codifferential(theta)
            momentum = C.cochain_mul(B, curvature)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            return energy

        jac = jit(grad(energy_elastica))

        # define linear constraint theta(0) = 0, delta_theta[-1] = 0
        cons = ({'type': 'eq', 'fun': lambda x: x[0]},
                {'type': 'eq', 'fun': lambda x: x[num_nodes-2] - x[num_nodes-1]})
        res = minimize(fun=energy_elastica, x0=theta_0, args=(E), method="SLSQP",
                       jac=jac, constraints=cons, options={'disp': 1, 'maxiter': 500})
        print(res)
        theta = res.x
        theta = theta[::density]

    # recover x and y
    x = np.empty(num_fem_elements + 1)
    y = np.empty(num_fem_elements + 1)
    x[0] = 0
    y[0] = 0
    for i in range(num_fem_elements):
        x[i + 1] = x[i] + h * np.cos(theta[i])
        y[i + 1] = y[i] + h * np.sin(theta[i])

    print(x)
    print(y)
    print(np.linalg.norm(x - x_true))
    print(np.linalg.norm(y - y_true))

    # plot the results
    plt.plot(x_true, y_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()


if __name__ == "__main__":
    # test_elastica(is_bilevel=True)
    test_elastica_data(data_type=0, is_bilevel=False)
