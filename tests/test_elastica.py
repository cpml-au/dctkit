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
from scipy import sparse
import matplotlib.pyplot as plt
import os

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


def get_elastica_mesh(num_fem_elements, L):
    # load simplicial complex
    num_nodes = num_fem_elements + 1
    S_1, x = util.generate_1_D_mesh(num_nodes, L)
    S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    # modify hodge star inverse at the boundary
    # S.hodge_star_inverse[0][0] /= 2
    # S.hodge_star_inverse[0][-1] /= 2
    return S


def test_elastica(is_bilevel=False):
    np.random.seed(42)

    # load FEM solution for benchmark
    filename = os.path.join(os.path.dirname(__file__), "theta_bench_FEM.txt")
    theta_true = np.genfromtxt(filename)[:, 1]

    num_elements_data = 100
    num_nodes_data = num_elements_data + 1
    density = 1
    num_elements = num_elements_data*density
    S = get_elastica_mesh(num_elements, 1)
    num_nodes = S.num_nodes

    # set params
    A = -4.

    # recover x_true, y_true
    x_true = np.empty(num_nodes_data)
    y_true = np.empty(num_nodes_data)
    x_true[0] = 0
    y_true[0] = 0
    h = 1/num_elements_data
    for i in range(num_elements_data):
        x_true[i + 1] = x_true[i] + h * np.cos(theta_true[i])
        y_true[i + 1] = y_true[i] + h * np.sin(theta_true[i])

    # cochain to handle boundary nodes
    internal_vec = np.ones(num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0
    internal_vec[-1] = 0
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    print(f"theta_true:{theta_true}")

    if is_bilevel:
        gamma = 10000
        theta_0 = 0.1*np.random.rand(num_elements).astype(dt.float_dtype)

        def energy_elastica_constr(theta: np.array, B: np.array) -> float:
            # theta = jnp.insert(theta, 0, 0)
            penalty = 0.5*gamma*dt.backend.sum((theta[0])**2)
            theta = C.CochainD0(complex=S, coeffs=theta)
            const = C.CochainD0(complex=S, coeffs=A *
                                np.ones(num_elements, dtype=dt.float_dtype))
            curvature = C.star(C.coboundary(theta))
            B_coch = C.scalar_mul(internal_coch, B[0])
            momentum = C.cochain_mul(curvature, B_coch)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            energy += penalty
            return energy

        def obj_fun(theta_guess: np.array, B_guess: float) -> float:
            # theta_guess = jnp.insert(theta_guess, 0, 0)
            return jnp.sum(jnp.square(theta_guess-theta_true[:-1]))
        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun, state_en=energy_elastica_constr, state_dim=num_elements)
        B_0 = 0.5*np.ones(1, dtype=dt.float_dtype)
        theta, B, fval = prb.run(theta_0, B_0, tol=1e-9)
        # extend theta
        # theta = np.insert(theta, 0, 0)
        print(f"The optimal B is {B[0]}")
        print(fval)
        # assert fval < 1e-3

    else:
        B = 1.
        theta_0 = 0.1*np.random.rand(num_elements).astype(dt.float_dtype)

        def energy_elastica(theta: np.array, B: float) -> float:
            theta = C.CochainD0(complex=S, coeffs=theta)
            const = C.CochainD0(complex=S, coeffs=A *
                                np.ones(num_elements, dtype=dt.float_dtype))
            curvature = C.star(C.coboundary(theta))
            B_coch = C.scalar_mul(internal_coch, B)
            momentum = C.cochain_mul(curvature, B_coch)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            return energy

        jac = jit(grad(energy_elastica))

        # define linear constraint theta(0) = 0, delta_theta[-1] = 0
        cons = ({'type': 'eq', 'fun': lambda x: x[0]})

        res = minimize(fun=energy_elastica, x0=theta_0, args=(B), method="SLSQP",
                       jac=jac, constraints=cons, options={'disp': 1, 'maxiter': 500})
        print(res)
        theta = res.x
        # print(f"theta_dual:{theta_dual}")

    print(f"theta:{theta}")

    # recover x and y
    x = np.empty(num_nodes)
    y = np.empty(num_nodes)
    x[0] = 0
    y[0] = 0
    h = 1/num_elements
    for i in range(num_elements):
        x[i + 1] = x[i] + h * np.cos(theta[i])
        y[i + 1] = y[i] + h * np.sin(theta[i])

    print(f"x:{x}")
    print(f"y:{y}")

    # plot momentum
    theta_coch = C.CochainD0(complex=S, coeffs=theta)
    curvature = C.star(C.coboundary(theta_coch))
    B_coch = C.scalar_mul(internal_coch, B)
    momentum = C.cochain_mul(curvature, B_coch)
    momentum.coeffs = momentum.coeffs.at[0].set(A*x[-1])
    print(f"momentum: {momentum.coeffs}")
    plt.plot(momentum.coeffs)
    plt.show()

    x = x[::density]
    y = y[::density]
    theta = theta[::density]

    # plot the results
    plt.plot(x_true, y_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()

    node_coord = S.node_coord[:, 0]
    node_coord = node_coord[::density]
    plt.plot(theta_true[:-1], 'r')
    plt.plot(theta, 'b')
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
    xi_true, eta_true = data_xy[:, 2*data_type], data_xy[:, 2*data_type + 1]

    density = 1
    num_elements_data = 10
    num_elements = num_elements_data*density
    R_0 = 1.2e-2
    R_1 = 0.3e-2
    L = 1.524
    # L = 1
    h = L/(num_elements)
    h_norm = h/L
    rho = R_1/R_0
    F = -data_F[data_type]
    S = get_elastica_mesh(num_elements, L)
    node_coord = S.node_coord[:, 0]
    num_nodes = S.num_nodes

    # define I_0
    I_0 = np.pi/4 * R_0**4
    # define I
    # I_node = I_0*(1 - (1 - rho)*node_coord/L)**4
    I_node = (1 - (1 - rho)*node_coord/L)**4
    # I = np.empty(density*num_fem_elements, dtype=dt.float_dtype)
    # for i in range(density*num_fem_elements):
    #    I[i] = 0.5*(I_node[i] + I_node[i+1])
    # I = I_0*(1 + 1/L*(1 - rho) * 1/num_fem_elements)**4 * \
    #    np.ones(num_fem_elements*density, dtype=dt.float_dtype)
    # I_coch = C.CochainP0(complex=S, coeffs=I)
    I_coch = C.CochainP0(complex=S, coeffs=I_node)

    # bidiagonal matrix
    diag = [1]*(num_nodes)
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    print(num_elements)
    print(transform.shape)

    # cochain to handle boundary nodes
    internal_vec = np.ones(num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0
    internal_vec[-1] = 0
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    if is_bilevel:
        gamma = 10000.
        theta_0 = 0.1*np.random.rand(num_elements).astype(dt.float_dtype)
        theta_true = np.empty(num_elements_data, dtype=dt.float_dtype)
        for i in range(num_elements_data):
            theta_true[i] = np.arctan(
                (eta_true[i+1]-eta_true[i])/(xi_true[i+1]-xi_true[i]))

        def energy_elastica_constr(theta: np.array, E: np.array) -> float:
            # theta = jnp.insert(theta, 0, 0)
            # define A
            # print(E[0])
            A = F
            penalty = 0.5*gamma*(theta[0])**2
            # jax.debug.print("{E}", E=E[0])
            # define B
            # B = C.CochainD0(complex=S, coeffs=E*C.star(C.coboundary(I_coch)).coeffs)
            B = C.CochainP0(complex=S, coeffs=I_coch.coeffs*(E[0]*I_0*1e9))
            B_in = C.cochain_mul(B, internal_coch)
            # get dimensionless B
            theta = C.CochainD0(complex=S, coeffs=theta)
            const = C.CochainD0(complex=S, coeffs=A *
                                np.ones(num_elements, dtype=dt.float_dtype))
            curvature = C.star(C.coboundary(theta))
            momentum = C.cochain_mul(B_in, curvature)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta)) + penalty
            return energy

        def obj_fun_explicit(theta_guess: np.array, E_guess: np.array) -> float:
            theta_guess = jnp.insert(theta_guess, 0, 0)
            # recover x and y
            cos_theta_guess = h_norm*jnp.cos(theta_guess)
            sin_theta_guess = h_norm*jnp.sin(theta_guess)
            b_xi = jnp.insert(cos_theta_guess, 0, 0)
            b_eta = jnp.insert(sin_theta_guess, 0, 0)
            xi = jnp.linalg.solve(transform, b_xi)
            eta = jnp.linalg.solve(transform, b_eta)
            xi = xi[::density]
            eta = eta[::density]
            return jnp.linalg.norm(xi-xi_true) + jnp.linalg.norm(eta-eta_true)

        def obj_fun_theta(theta_guess: np.array, B_guess: np.array) -> float:
            # theta_guess = jnp.insert(theta_guess, 0, 0)
            # theta_guess = theta_guess[::density]
            obj = jnp.sum(jnp.square(theta_guess-theta_true))
            jax.debug.print("{obj}", obj=obj)
            return obj

        prb = optctrl.OptimalControlProblem(
            objfun=obj_fun_theta, state_en=energy_elastica_constr, state_dim=num_elements)
        E_0 = 0.01*np.ones(1, dtype=dt.float_dtype)
        theta, E, fval = prb.run(theta_0, E_0, tol=1e-2)
        # extend theta
        # theta = np.insert(theta, 0, 0)
        plt.plot(theta)
        plt.show()
        print(f"The optimal E is {E[0]}*10^9")
        print(f"fval: {fval}")
        # assert fval < 1e-3

    else:
        E_all = np.array([6.e9, 5.6e9, 6.5e9, 5.8e9])
        E = E_all[data_type]
        print(f"E = {E}")
        theta_0 = 0.1*np.random.rand(num_elements).astype(dt.float_dtype)
        A = F
        print(f"A = {A}")

        def energy_elastica(theta: np.array, E: float) -> float:
            # define A
            # A = F*L**2/(E*I_0)
            # define B
            # B = C.CochainD0(complex=S, coeffs=E*C.star(C.coboundary(I_coch)).coeffs)
            B = C.CochainP0(complex=S, coeffs=I_coch.coeffs*(E*I_0))
            B_in = C.cochain_mul(B, internal_coch)
            # B = C.CochainD0(complex=S, coeffs=np.ones(density*num_fem_elements))
            # get dimensionless B
            # B = C.scalar_mul(B, L**2/(E*I_0))
            # print(B.coeffs)
            theta = C.CochainD0(complex=S, coeffs=theta)
            const = C.CochainD0(complex=S, coeffs=A *
                                np.ones(num_elements, dtype=dt.float_dtype))
            curvature = C.star(C.coboundary(theta))
            momentum = C.cochain_mul(B_in, curvature)
            energy = 0.5*C.inner_product(momentum, curvature) - \
                C.inner_product(const, C.sin(theta))
            return energy

        jac = jit(grad(energy_elastica))

        # define linear constraint theta(0) = 0, delta_theta[-1] = 0
        cons = ({'type': 'eq', 'fun': lambda x: x[0]})
        res = minimize(fun=energy_elastica, x0=theta_0, args=(E), method="SLSQP",
                       jac=jac, constraints=cons, options={'disp': 1, 'maxiter': 500})
        print(res)
        theta = res.x

    plt.plot(theta)
    plt.show()
    '''
        curvature = C.codifferential(C.CochainD1(S, theta))
        B = C.CochainD0(complex=S, coeffs=E*I_coch.coeffs)
        momentum = C.cochain_mul(B, curvature)
        plt.plot(B.coeffs)
        plt.show()
        print(f"theta={theta}")
        plt.plot(momentum.coeffs*L/(num_fem_elements*density))
        plt.show()
        '''

    cos_theta = h*jnp.cos(theta)
    sin_theta = h*jnp.sin(theta)
    b_xi = jnp.insert(cos_theta, 0, 0)
    b_eta = jnp.insert(sin_theta, 0, 0)
    xi = jnp.linalg.solve(transform, b_xi)
    eta = jnp.linalg.solve(transform, b_eta)
    xi = xi[::density]
    eta = eta[::density]
    '''
    xi = np.empty(num_nodes)
    eta = np.empty(num_nodes)
    xi[0] = 0
    eta[0] = 0
    for i in range(num_elements):
        xi[i + 1] = xi[i] + h_norm * np.cos(theta[i])
        eta[i + 1] = eta[i] + h_norm * np.sin(theta[i])
    print(f"xi: {xi}")
    print(f"eta: {eta}")
    xi = xi[::density]
    eta = eta[::density]
    '''

    # get x and y
    x = L*xi
    y = L*eta
    print(x)
    print(y)

    # get x_true and y_true
    x_true = L*xi_true
    y_true = L*eta_true

    print(xi_true)
    print(eta_true)
    print(np.linalg.norm(xi - xi_true))
    print(np.linalg.norm(eta - eta_true))

    # plot the results
    plt.plot(y_true, -x_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()


if __name__ == "__main__":
    # test_elastica(is_bilevel=True)
    test_elastica_data(data_type=0, is_bilevel=True)
