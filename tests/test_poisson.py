import numpy as np
from scipy.optimize import minimize
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


cwd = os.path.dirname(simplex.__file__)


def get_complex(S_p, node_coords, float_dtype="float64", int_dtype="int64"):
    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    if type != "float64":
        bnodes = np.array(bnodes, dtype=np.int32)
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_p, node_coords, float_dtype, int_dtype)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    return S, bnodes, triang


def test_poisson(energy_bool=True, optimizer="jaxopt", float_dtype="float64",
                 int_dtype="int64"):

    # tested with test1.msh, test2.msh and test3.msh

    # filename = "test1.msh"
    # full_path = os.path.join(cwd, filename)
    gmsh.initialize()
    history = []
    history_boundary = []
    final_energy = []
    lc = 1.0
    j = 15
    for i in range(j):
        print("i = ", i)
        _, _, S_2, node_coord = util.generate_mesh(lc)

        # numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

        S, bnodes, triang = get_complex(S_2, node_coord, float_dtype, int_dtype)
        # TODO: initialize diffusivity
        k = 1.

        # exact solution
        u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1]**2, dtype=float_dtype)
        b_values = u_true[bnodes]
        '''
        plt.tricontourf(triang, u_true, cmap='RdBu', levels=20)
        plt.triplot(triang, 'ko-')
        plt.colorbar()
        plt.show()
        '''
        # TODO: initialize boundary_values
        boundary_values = (np.array(bnodes), b_values)
        # TODO: initialize external sources
        dim_0 = S.num_nodes
        f_vec = 4.*np.ones(dim_0, dtype=float_dtype)

        mask = np.ones(dim_0, dtype=float_dtype)
        mask[bnodes] = 0.

        if energy_bool:
            obj = p.energy_poisson
            gradfun = p.grad_energy_poisson
            gamma = 1000.
            args = (f_vec, S, k, boundary_values, gamma)
        else:
            obj = p.obj_poisson
            gradfun = p.grad_poisson
            f = C.Cochain(0, True, S, f_vec)
            star_f = C.star(f)
            # penalty factor on boundary conditions
            gamma = 10000.
            args = (star_f.coeffs, S, k, boundary_values, gamma, mask)

        # initial guess
        u_0 = 0.01*np.random.rand(dim_0)

        if optimizer == "scipy":
            u = minimize(fun=obj, x0=u_0, args=args, method='BFGS',
                         jac=gradfun, options={'disp': 1})

            plt.tricontourf(triang, u.x, cmap='RdBu', levels=20)
            plt.triplot(triang, 'ko-')
            plt.colorbar()
            plt.show()
            x = u.x
            minf = u.fun
            current_history = np.linalg.norm(x-u_true)
            current_history_boundary = np.linalg.norm(x[bnodes]-u_true[bnodes])

        elif optimizer == "nlopt":
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
            # opt = nlopt.opt(nlopt.LD_SLSQP, dim_0)
            # opt.set_lower_bounds([-float('inf'), 0])

            # Set objective function to minimize
            opt.set_min_objective(f2)

            opt.set_ftol_abs(1e-8)
            xinit = u_0

            tic = time.time()
            x = opt.optimize(xinit)
            toc = time.time()

            minf = opt.last_optimum_value()
            print("minimum value = ", minf)
            print("result code = ", opt.last_optimize_result())
            print("Elapsed time = ", toc - tic)
            current_history = np.linalg.norm(x-u_true)
            current_history_boundary = np.linalg.norm(x[bnodes]-u_true[bnodes])

        elif optimizer == "jaxopt":
            sol_true = jnp.array(u_true)
            u_0 = jnp.array(u_0)
            f_vec = jnp.array(f_vec)
            bnodes = jnp.array(bnodes)
            b_values = jnp.array(b_values)
            boundary_values = (bnodes, b_values)

            if energy_bool:
                def energy_poisson(x, f, k, boundary_values, gamma):
                    pos, value = boundary_values
                    f = C.Cochain(0, True, S, f, "jax")
                    u = C.Cochain(0, True, S, x, "jax")
                    du = C.coboundary(u)
                    norm_grad = k/2*C.inner_product(du, du)
                    bound_term = C.inner_product(u, f)
                    penalty = 0.5*gamma*jnp.sum((x[pos] - value)**2)
                    energy = norm_grad + bound_term + penalty
                    return energy

                new_args = (f_vec, k, boundary_values, gamma)
                obj = energy_poisson

            else:
                def obj_poisson(x, f, k, boundary_values, gamma, mask):
                    # f, k, boundary_values, gamma, mask = tuple
                    pos, value = boundary_values
                    Ax = p.poisson_vec_operator(x, S, k, "jax")
                    r = Ax - f
                    # zero residual on dual cells at the boundary where nodal values are
                    # imposed

                    # \sum_i (x_i - value_i)^2
                    penalty = jnp.sum((x[pos] - value)**2)
                    energy = 0.5*jnp.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
                    return energy

                new_args = (star_f.coeffs, k, boundary_values, gamma, mask)
                obj = obj_poisson

            solver = jaxopt.LBFGS(obj, maxiter=5000)
            tic = time.time()
            sol = solver.run(u_0, *new_args)
            toc = time.time()
            print("Elapsed time = ", toc-tic)
            print(sol.state.iter_num, sol.state.value,
                  jnp.linalg.norm(sol.params[bnodes]-sol_true[bnodes]))
            x = sol.params
            minf = sol.state.value
            current_history = jnp.linalg.norm(x-sol_true)
            current_history_boundary = jnp.linalg.norm(x[bnodes]-sol_true[bnodes])

        history.append(current_history)
        history_boundary.append(current_history_boundary)
        final_energy.append(minf)
        lc = lc/np.sqrt(2)

        # assert np.allclose(u.x[bnodes], u_true[bnodes], atol=1e-6)
        # assert np.allclose(u.x, u_true, atol=1e-6)

    plt.plot(range(j), history, label="Error")
    plt.plot(range(j), history_boundary, label="Boundary Error")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(range(j), final_energy, label="Final Energy")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    test_poisson(True, "jaxopt", "float32", "int32")
