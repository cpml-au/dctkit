import numpy as np
from scipy.optimize import minimize
import dctkit
from dctkit import config
from dctkit.mesh import simplex, util
from dctkit.apps import poisson as p
from dctkit.dec import cochain as C
import os
import matplotlib.tri as tri
import nlopt
import jax.numpy as jnp
import jaxopt
import gmsh

cwd = os.path.dirname(simplex.__file__)


def get_complex(S_p, node_coords):
    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    bnodes = bnodes.astype(dctkit.int_dtype)
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_p, node_coords, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    return S, bnodes, triang


def test_poisson(energy_formulation=False, optimizer="jaxopt", float_dtype=dctkit.FloatDtype.float32, int_dtype=dctkit.IntDtype.int32):

    config(fdtype=float_dtype, idtype=int_dtype)

    # tested with test1.msh, test2.msh and test3.msh

    np.random.seed(42)

    lc = 0.5

    _, _, S_2, node_coord = util.generate_square_mesh(lc)

    S, bnodes, triang = get_complex(S_2, node_coord)

    k = 1.

    # NOTE: exact solution of -Delta u + f = 0
    u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1]
                      ** 2, dtype=dctkit.float_dtype)
    b_values = u_true[bnodes]

    boundary_values = (np.array(bnodes, dtype=dctkit.int_dtype), b_values)

    dim_0 = S.num_nodes
    f_vec = -4.*np.ones(dim_0, dtype=dctkit.float_dtype)
    f = C.Cochain(0, True, S, f_vec)
    star_f = C.star(f)

    mask = np.ones(dim_0, dtype=dctkit.float_dtype)
    mask[bnodes] = 0.

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0)
    u_0 = np.array(u_0, dtype=dctkit.float_dtype)

    if optimizer == "scipy":
        print("Using SciPy optimizer...")

        if energy_formulation:
            print("Using energy formulation...")
            obj = p.energy_poisson
            gradfun = p.grad_energy_poisson
            gamma = 1000.
            args = (f_vec, S, k, boundary_values, gamma)
        else:
            print("Solving Poisson equation...")
            obj = p.obj_poisson
            gradfun = p.grad_poisson
            gamma = 10000.
            args = (star_f.coeffs, S, k, boundary_values, gamma, mask)

        u = minimize(fun=obj, x0=u_0, args=args, method='BFGS',
                     jac=gradfun, options={'disp': 1})
        u = u.x

    elif optimizer == "nlopt":
        print("Using NLOpt optimizer (only energy formulation)...")
        obj = p.energy_poisson
        gradfun = p.grad_energy_poisson
        gamma = 1000.

        def f2(x, grad):
            if grad.size > 0:
                grad[:] = gradfun(x, f_vec, S, k,
                                  boundary_values, gamma)

            return np.double(obj(x, f_vec, S, k, boundary_values, gamma))
            # NOTE: this casting to double is crucial to work with NLOpt
            # return np.double(fjax(x))

        # The second argument is the number of optimization parameters
        opt = nlopt.opt(nlopt.LD_LBFGS, dim_0)

        # Set objective function to minimize
        opt.set_min_objective(f2)

        opt.set_ftol_abs(1e-8)
        xinit = u_0

        u = opt.optimize(xinit)

    elif optimizer == "jaxopt":
        print("Using jaxopt optimizer...")

        gamma = 1000.

        if energy_formulation:
            print("Using energy formulation...")

            def energy_poisson(x, f, k, boundary_values, gamma):
                pos, value = boundary_values
                f = C.Cochain(0, True, S, f, "jax")
                u = C.Cochain(0, True, S, x, "jax")
                du = C.coboundary(u)
                norm_grad = k/2.*C.inner_product(du, du)
                bound_term = C.inner_product(u, f)
                penalty = 0.5*gamma*jnp.sum((x[pos] - value)**2)
                energy = norm_grad + bound_term + penalty
                return energy

            new_args = (f_vec, k, boundary_values, gamma)
            obj = energy_poisson

        else:
            print("Solving Poisson equation...")

            def obj_poisson(x, f, k, boundary_values, gamma, mask):
                # f, k, boundary_values, gamma, mask = tuple
                pos, value = boundary_values
                Ax = p.poisson_vec_operator(x, S, k, "jax")
                r = Ax + f
                # zero residual on dual cells at the boundary where nodal values are
                # imposed

                # \sum_i (x_i - value_i)^2
                penalty = jnp.sum((x[pos] - value)**2)
                energy = 0.5*jnp.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
                return energy

            def obj_laplacian_poisson(x, f, k, boundary_values, gamma, mask):
                pos, value = boundary_values
                c = C.Cochain(0, True, S, x, type)
                # define laplacian
                laplacian = C.laplacian(c)
                # laplacian on forms is minus laplacian on vector fields
                laplacian.coeffs *= -k
                # proceed as obj_poisson
                r = laplacian.coeffs + f
                penalty = jnp.sum((x[pos] - value)**2)
                energy = 0.5*jnp.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
                return energy

            star_f.type = "jax"
            new_args = (star_f.coeffs, k, boundary_values, gamma, mask)
            # since to build laplacian we must do another star, we have to do another
            # star also to star_f
            new_lap_args = (C.star(star_f).coeffs, k, boundary_values, gamma, mask)
            obj = obj_poisson
            obj_lap = obj_laplacian_poisson

            solver_lap = jaxopt.LBFGS(obj_lap, maxiter=5000)
            sol_lap = solver_lap.run(u_0, *new_lap_args)
            u_lap = sol_lap.params

            assert np.allclose(u_lap[bnodes], u_true[bnodes], atol=1e-2)
            assert np.allclose(u_lap, u_true, atol=1e-2)

        solver = jaxopt.LBFGS(obj, maxiter=5000)
        sol = solver.run(u_0, *new_args)
        u = sol.params

    assert u.dtype == dctkit.float_dtype
    assert u_true.dtype == u.dtype
    assert np.allclose(u[bnodes], u_true[bnodes], atol=1e-2)
    assert np.allclose(u, u_true, atol=1e-2)


if __name__ == "__main__":
    test_poisson(float_dtype=dctkit.FloatDtype.float32, int_dtype=dctkit.IntDtype.int32)
    test_poisson(float_dtype=dctkit.FloatDtype.float64, int_dtype=dctkit.IntDtype.int64)
