import numpy as np
import jax.numpy as jnp
import jax
import dctkit as dt
from scipy.optimize import minimize
from dctkit.mesh import simplex, util
from dctkit.physics import poisson as p
from dctkit.dec import cochain as C
from dctkit.math.opt import optctrl as oc
import matplotlib.tri as tri
import jaxopt
import gmsh
import pytest
from functools import partial


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


cases = [["jaxopt", False], ["jaxopt", True], ["pygmo", True],
         ["pygmo", False], ["scipy", False], ["scipy", True]]


@pytest.mark.parametrize('optimizer,energy_formulation', cases)
def test_poisson(setup_test, optimizer, energy_formulation):
    """Solve the problem k*Delta u + f = 0."""
    if jax.config.read("jax_enable_x64"):
        assert dt.float_dtype == "float64"

    np.random.seed(42)

    lc = 0.5

    util.generate_square_mesh(lc)
    _, _, S_2, node_coord, _ = util.read_mesh()

    S, bnodes, _ = get_complex(S_2, node_coord)

    k = 1.

    # NOTE: exact solution of Delta u + f = 0
    u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1]
                      ** 2, dtype=dt.float_dtype)
    b_values = u_true[bnodes]

    boundary_values = (np.array(bnodes, dtype=dt.int_dtype), b_values)

    dim_0 = S.num_nodes
    f_vec = -4.*np.ones(dim_0, dtype=dt.float_dtype)
    f = C.Cochain(0, True, S, f_vec)
    star_f = C.star(f)

    mask = np.ones(dim_0, dtype=dt.float_dtype)
    mask[bnodes] = 0.

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0).astype(dt.float_dtype)
    u_0 = np.array(u_0, dtype=dt.float_dtype)

    if optimizer == "scipy":
        print("Using SciPy optimizer...")

        if energy_formulation:
            print("Using energy formulation...")
            obj = p.energy_poisson
            gradfun = p.grad_energy_poisson
            gamma = 1e3
            args = (f_vec, S, k, boundary_values, gamma)
        else:
            print("Solving Poisson equation...")
            obj = p.obj_poisson
            gradfun = p.grad_obj_poisson
            gamma = 10000.
            args = (star_f.coeffs, S, k, boundary_values, gamma, mask)

        u = minimize(fun=obj, x0=u_0, args=args, method='BFGS',
                     jac=gradfun, options={'disp': 1})
        # NOTE: minimize returns a float64 array
        u = u.x.astype(dt.float_dtype)

    elif optimizer == "pygmo":
        print("Using pygmo optimizer...")

        gamma = 1000.
        if energy_formulation:
            print("Using energy formulation...")
            obj = partial(p.energy_poisson, S=S)
            args = {'f': f_vec, 'k': k, 'boundary_values': boundary_values,
                    'gamma': gamma}

        else:
            print("Solving Poisson equation...")

            def obj_poisson(x, f, k, boundary_values, gamma, mask):
                pos, value = boundary_values
                c = C.Cochain(0, True, S, x)
                # compute Laplace-de Rham of c
                laplacian = C.laplacian(c)
                # the Laplacian on forms is the negative of the Laplacian on scalar
                # fields
                laplacian.coeffs *= -k
                # compute the residual of the Poisson equation k*Delta u + f = 0
                r = laplacian.coeffs + f
                penalty = jnp.sum((x[pos] - value)**2)
                obj = 0.5*jnp.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
                return obj
            obj = obj_poisson
            args = {'f': f_vec, 'k': k, 'boundary_values': boundary_values,
                    'gamma': gamma, 'mask': mask}

        prb = oc.OptimizationProblem(dim=dim_0, state_dim=dim_0, objfun=obj)
        prb.set_obj_args(args)
        u = prb.run(u_0, algo="lbfgs").astype(dt.float_dtype)

    elif optimizer == "jaxopt":
        print("Using jaxopt optimizer...")

        gamma = 1000.

        if energy_formulation:
            print("Using energy formulation...")

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

        else:
            print("Solving Poisson equation...")

            def obj_poisson(x, f, k, boundary_values, gamma, mask):
                pos, value = boundary_values
                c = C.Cochain(0, True, S, x)
                # compute Laplace-de Rham of c
                laplacian = C.laplacian(c)
                # the Laplacian on forms is the negative of the Laplacian on scalar
                # fields
                laplacian.coeffs *= -k
                # compute the residual of the Poisson equation k*Delta u + f = 0
                r = laplacian.coeffs + f
                penalty = jnp.sum((x[pos] - value)**2)
                obj = 0.5*jnp.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
                return obj

            args = (f_vec, k, boundary_values, gamma, mask)
            obj = obj_poisson

        solver = jaxopt.LBFGS(obj, maxiter=5000)
        sol = solver.run(u_0, *args)
        u = sol.params

    assert u.dtype == dt.float_dtype
    assert u_true.dtype == u.dtype
    assert np.allclose(u[bnodes], u_true[bnodes], atol=1e-2)
    assert np.allclose(u, u_true, atol=1e-2)
