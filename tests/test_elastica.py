import numpy as np
import dctkit as dt
import jax.numpy as jnp
from jax import grad, Array
from dctkit.math.opt import optctrl
from scipy import sparse
import os
from dctkit.apps import elastica as el
import numpy.typing as npt
from dctkit.dec import cochain as C
# import pytest


# @pytest.mark.parametrize('tune_EI0', [True, False])
def test_elastica_energy(setup_test):
    data = "data/xy_F_20.txt"
    F = -20
    np.random.seed(42)

    # load true data
    filename = os.path.join(os.path.dirname(__file__), data)
    data = np.genfromtxt(filename)

    # sampling factor for true data
    sampling = 10

    num_elements = 10

    L = 1
    h = L/(num_elements)

    ela = el.ElasticaProblem(num_elements=num_elements, L=L, rho=1.)
    num_nodes = ela.S.num_nodes

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(num_nodes)
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # initial guess for the angles (EXCEPT THE FIRST ANGLE, FIXED BY BC)
    theta_0 = 0.1*np.random.rand(num_elements-1).astype(dt.float_dtype)

    # compute true solution
    x_true = data[:, 1][::sampling]
    y_true = data[:, 2][::sampling]
    theta_true = np.empty(num_elements, dtype=dt.float_dtype)
    for i in range(num_elements):
        theta_true[i] = np.arctan(
            (y_true[i+1]-y_true[i])/(x_true[i+1]-x_true[i]))

    # state function: stationarity conditions of the elastic energy
    def statefun(x: npt.NDArray, theta_0: npt.NDArray, F: float) -> Array:
        u = x[:-1]
        EI0 = x[-1]
        return grad(ela.energy_elastica)(u, EI0, theta_0, F)

    # define extra_args
    constraint_args = {'theta_0': theta_true[0], 'F': F}
    obj_args = {'theta_true': theta_true}

    def obj(x: npt.NDArray, theta_true: npt.NDArray) -> Array:
        theta_guess = x[:-1]
        EI_guess = x[-1]
        return ela.obj_fun_theta(theta_guess, EI_guess, theta_true)

    prb = optctrl.OptimalControlProblem(objfun=obj,
                                        statefun=statefun,
                                        state_dim=num_elements-1,
                                        nparams=num_elements,
                                        constraint_args=constraint_args,
                                        obj_args=obj_args)
    EI0_0 = 1*np.ones(1, dtype=dt.float_dtype)
    x0 = np.concatenate((theta_0, EI0_0))
    x = prb.run(x0=x0)
    theta = x[:-1]
    # extend solution array with boundary element
    theta = np.insert(theta, 0, theta_true[0])

    # reconstruct x, y
    cos_theta = h*jnp.cos(theta)
    sin_theta = h*jnp.sin(theta)
    b_x = jnp.insert(cos_theta, 0, 0)
    b_y = jnp.insert(sin_theta, 0, 0)
    x = jnp.linalg.solve(transform, b_x)
    y = jnp.linalg.solve(transform, b_y)

    error = np.linalg.norm(x - x_true) + np.linalg.norm(y - y_true)
    assert error <= 2e-2


def test_elastica_equation(setup_test):
    data = "data/xy_F_10.txt"
    F = -10.
    np.random.seed(42)

    # load true data
    filename = os.path.join(os.path.dirname(__file__), data)
    data = np.genfromtxt(filename)

    # sampling factor for true data
    sampling = 10

    num_elements = 10

    L = 1
    h = L/(num_elements)

    B = 7.854

    penalty = 10.

    ela = el.ElasticaProblem(num_elements=num_elements, L=L, rho=1.)
    num_nodes = ela.S.num_nodes

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(num_nodes)
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # compute true solution
    x_true = data[:, 1][::sampling]
    y_true = data[:, 2][::sampling]
    theta_true = np.empty(num_elements, dtype=dt.float_dtype)
    for i in range(num_elements):
        theta_true[i] = np.arctan(
            (y_true[i+1]-y_true[i])/(x_true[i+1]-x_true[i]))

    # initial guess for the angles
    theta_0 = theta_true[0]*np.zeros(num_elements, dtype=dt.float_dtype)

    # internal cochain
    intc = np.ones(num_nodes, dtype=dt.float_dtype)
    intc[0] = 0.
    intc[-1] = 0.
    int_coch = C.CochainP0(ela.S, intc)

    mask = np.ones(len(theta_0), dtype=dt.float_dtype)
    mask[0] = 0.
    mask_coch = C.CochainP1(ela.S, mask)

    def obj(x: npt.NDArray) -> Array:
        theta = C.CochainD0(ela.S, x)
        dtheta = C.coboundary(theta)
        # primal 0-cochain
        curv = C.cochain_mul(int_coch, C.star(dtheta))
        # primal 1-cochain
        load = C.scalar_mul(C.star(C.cos(theta)), F)

        residual = C.add(C.coboundary(C.scalar_mul(curv, B)), load)
        mask_residual = C.cochain_mul(mask_coch, residual)
        penalty_term = penalty*(theta.coeffs[0] - theta_true[0])**2
        return C.inner_product(mask_residual, mask_residual) + penalty_term

    prb = optctrl.OptimizationProblem(
        dim=num_elements, state_dim=num_elements, objfun=obj)

    prb.set_obj_args({})
    theta = prb.run(x0=theta_0)

    # reconstruct x, y
    cos_theta = h*jnp.cos(theta)
    sin_theta = h*jnp.sin(theta)
    b_x = jnp.insert(cos_theta, 0, 0)
    b_y = jnp.insert(sin_theta, 0, 0)
    x = jnp.linalg.solve(transform, b_x)
    y = jnp.linalg.solve(transform, b_y)

    error = np.linalg.norm(x - x_true) + np.linalg.norm(y - y_true)
    print(theta)
    print(theta_true)
    print(prb.last_opt_result)
    assert error <= 2e-2
