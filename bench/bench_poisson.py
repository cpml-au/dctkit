import numpy as np
from scipy.optimize import minimize
import dctkit as dt
from dctkit.mesh import simplex, util
from dctkit.apps import poisson as p
from dctkit.dec import cochain as C
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import nlopt
import time
import jax.numpy as jnp
import jaxopt
import gmsh
import jax

from dctkit import config, FloatDtype, IntDtype, Backend, Platform

cwd = os.path.dirname(simplex.__file__)


def get_complex(S_p, node_coords):
    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    bnodes = bnodes.astype(dt.int_dtype)
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_p, node_coords, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    return S, bnodes, triang


def bench_poisson(optimizer="scipy", float_dtype="float32", int_dtype="int32"):

    # NOTE: NLOpt only works with float64
    config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.cpu)

    if jax.config.read("jax_enable_x64"):
        assert dt.float_dtype == "float64"

    np.random.seed(42)
    lc = 0.1

    _, _, S_2, node_coord = util.generate_square_mesh(lc)
    S, bnodes, _ = get_complex(S_2, node_coord)

    obj = p.energy_poisson
    gradfun = p.grad_energy_poisson
    gamma = 1000.

    # NOTE: exact solution of Delta u + f = 0
    u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1]
                      ** 2, dtype=dt.float_dtype)
    b_values = u_true[bnodes]

    boundary_values = (np.array(bnodes, dtype=dt.int_dtype), b_values)

    k = 1.

    dim_0 = S.num_nodes
    f_vec = -4.*np.ones(dim_0, dtype=dt.float_dtype)

    mask = np.ones(dim_0, dtype=dt.float_dtype)
    mask[bnodes] = 0.

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0).astype(dt.float_dtype)

    args = (f_vec, S, k, boundary_values, gamma)

    tic = time.time()
    if optimizer == "scipy":
        print("Using SciPy optimizer...")
        res = minimize(fun=obj, x0=u_0, args=args, method='BFGS',
                       jac=gradfun, options={'disp': 1})

        # NOTE: minimize returns a float64 array
        u = res.x.astype(dt.float_dtype)
        minf = res.fun
        toc = time.time()

    elif optimizer == "nlopt":
        print("Using NLOpt optimizer...")
        obj = p.energy_poisson
        gradfun = p.grad_energy_poisson

        def f2(x, grad):
            if grad.size > 0:
                grad[:] = gradfun(x, f_vec, S, k,
                                  boundary_values, gamma)

            return np.double(obj(x, f_vec, S, k, boundary_values, gamma))
        # NOTE: this casting to double is crucial to work with NLOpt
        # return np.double(fjax(x))

        # The second argument is the number of optimization parameters
        opt = nlopt.opt(nlopt.LD_LBFGS, dim_0)
        # opt = nlopt.opt(nlopt.LD_SLSQP, dim_0)
        # opt.set_lower_bounds([-float('inf'), 0])

        # Set objective function to minimize
        opt.set_min_objective(f2)

        opt.set_ftol_abs(1e-8)
        xinit = u_0

        u = opt.optimize(xinit)
        toc = time.time()

        minf = opt.last_optimum_value()
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())

    elif optimizer == "jaxopt":
        print("Using jaxopt optimizer...")

        gamma = 1000.

        def energy_poisson(x, f, k, boundary_values, gamma):
            pos, value = boundary_values
            f = C.Cochain(0, True, S, f)
            u = C.Cochain(0, True, S, x)
            du = C.coboundary(u)
            norm_grad = k/2.*C.inner_product(du, du)
            bound_term = -C.inner_product(u, f)
            penalty = 0.5*gamma*dt.backend.sum((x[pos] - value)**2)
            energy = norm_grad + bound_term + penalty
            return energy

        args = (f_vec, k, boundary_values, gamma)
        obj = energy_poisson

        solver = jaxopt.LBFGS(obj, maxiter=5000)
        sol = solver.run(u_0, *args)
        toc = time.time()
        print(sol.state.iter_num, sol.state.value,
              jnp.linalg.norm(sol.params[bnodes]-u_true[bnodes]))
        u = sol.params
        minf = sol.state.value

    print("Elapsed time = ", toc-tic)
    assert np.allclose(u[bnodes], u_true[bnodes], atol=1e-3)
    assert np.allclose(u, u_true, atol=1e-3)


if __name__ == '__main__':
    bench_poisson(optimizer="jaxopt")
    bench_poisson(optimizer="nlopt")
    bench_poisson(optimizer="scipy")
