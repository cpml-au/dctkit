import numpy as np
import dctkit as dt
import jax.numpy as jnp
from jax import jit, grad
from dctkit.math.opt import optctrl
from scipy.optimize import minimize
from scipy import sparse
import matplotlib.pyplot as plt
import os
from dctkit.apps import elastica as el


def test_elastica(setup_test):
    data = "xy_F_20.txt"
    F = -20
    is_bilevel = True
    np.random.seed(42)

    # load true data
    filename = os.path.join(os.path.dirname(__file__), data)
    data = np.genfromtxt(filename)

    # sample true data
    sampling = 10

    density = 1
    num_elements_data = int(100/sampling)
    num_elements = num_elements_data*density
    noise = 0.01*np.random.rand(num_elements_data+1)
    x_true = data[:, 1][::sampling] + noise
    y_true = data[:, 2][::sampling] + noise

    R_0 = 1e-2
    R_1 = 1e-2
    L = 1
    h = L/(num_elements)
    rho = R_1/R_0

    ela = el.ElasticaProblem(num_elements=num_elements, L=L, rho=rho)
    num_nodes = ela.S.num_nodes

    # define I_0
    I_0 = np.pi/4 * R_0**4

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(num_nodes)
    upper_diag = [-1]*(num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # get theta_true
    theta_true = np.empty(num_elements_data, dtype=dt.float_dtype)
    for i in range(num_elements_data):
        theta_true[i] = np.arctan(
            (y_true[i+1]-y_true[i])/(x_true[i+1]-x_true[i]))

    if is_bilevel:
        theta_0 = 0.1*np.random.rand(num_elements-1).astype(dt.float_dtype)

        # define extra_args
        constraint_args = (theta_true[0], F)
        obj_args = (theta_true)
        energy = ela.energy_elastica
        obj = ela.obj_fun_theta

        prb = optctrl.OptimalControlProblem(objfun=obj,
                                            state_en=energy,
                                            state_dim=num_elements-1,
                                            constraint_args=constraint_args,
                                            obj_args=obj_args)
        EI0_0 = 1*np.ones(1, dtype=dt.float_dtype)
        theta, EI0, fval = prb.run(theta_0, EI0_0, tol=1e-5)
        # extend theta
        theta = np.insert(theta, 0, theta_true[0])
        print(f"The optimal E*I_0 is {EI0[0]}")
        print(f"The optimal E is {EI0[0]/I_0}")
        print(f"fval: {fval}")
        # assert fval < 1e-3

    else:
        EI0 = [5.9856, 6.6421, 6.8612, 6.9562]
        EI0 = EI0[int(-F/5)-1]
        theta_0 = 0.1*np.random.rand(num_elements-1).astype(dt.float_dtype)
        energy_elastica = ela.energy_elastica

        jac = jit(grad(energy_elastica))

        res = minimize(fun=energy_elastica, x0=theta_0, args=(
            EI0, theta_true[0], F), method="SLSQP", jac=jac,
            options={'disp': 1, 'maxiter': 500})
        print(res)
        theta = res.x
        theta = np.insert(theta, 0, theta_true[0])

    # plot theta
    plt.plot(theta)
    plt.show()

    # reconstruct x, y
    cos_theta = h*jnp.cos(theta)
    sin_theta = h*jnp.sin(theta)
    b_x = jnp.insert(cos_theta, 0, 0)
    b_y = jnp.insert(sin_theta, 0, 0)
    x = jnp.linalg.solve(transform, b_x)
    y = jnp.linalg.solve(transform, b_y)
    x = x[::density]
    y = y[::density]

    # plot the results
    plt.plot(x_true, y_true, 'r')
    plt.plot(x, y, 'b')
    plt.show()

    error = np.linalg.norm(x - x_true) + np.linalg.norm(y - y_true)
    assert error < 5e-2
