import numpy as np
from scipy.optimize import minimize
import dctkit as dt
from dctkit.mesh import simplex, util
from dctkit.apps import poisson as p
import os
import sys
import matplotlib.tri as tri
import time
import jax.numpy as jnp
import jaxopt
import gmsh
import jax
import pygmo as pg
from functools import partial

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


def bench_poisson(optimizer="scipy", platform="cpu", float_dtype="float32",
                  int_dtype="int32"):

    # NOTE: NLOpt only works with float64
    if platform == "cpu":
        config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.cpu)
    else:
        config(FloatDtype.float64, IntDtype.int64, Backend.jax, Platform.gpu)

    if jax.config.read("jax_enable_x64"):
        assert dt.float_dtype == "float64"

    np.random.seed(42)
    lc = 0.05

    _, _, S_2, node_coord = util.generate_square_mesh(lc)
    S, bnodes, _ = get_complex(S_2, node_coord)

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

    # Dirichlet energy and its gradient (computed using JAX's autodiff)
    energy = jax.jit(partial(p.energy_poisson, f=f_vec, S=S,
                             k=k, boundary_values=boundary_values, gamma=gamma))
    graden = jax.jit(jax.grad(partial(p.energy_poisson, f=f_vec, S=S,
                                      k=k, boundary_values=boundary_values,
                                      gamma=gamma)))

    tic = time.time()
    if optimizer == "scipy":
        print("Using SciPy optimizer...")

        graden = partial(p.grad_energy_poisson, f=f_vec, S=S, k=k,
                         boundary_values=boundary_values, gamma=gamma)
        res = minimize(fun=energy, x0=u_0, method='BFGS',
                       jac=graden, options={'disp': 1})

        # NOTE: minimize returns a float64 array
        u = res.x.astype(dt.float_dtype)
        minf = res.fun
        toc = time.time()

    elif optimizer == "jaxopt":
        print("Using jaxopt optimizer...")

        solver = jaxopt.LBFGS(energy, maxiter=5000)
        sol = solver.run(u_0)
        toc = time.time()
        print(sol.state.iter_num, sol.state.value,
              jnp.linalg.norm(sol.params[bnodes]-u_true[bnodes]))
        u = sol.params

    elif optimizer == "pygmo":

        class PoissonProblem():
            def fitness(self, dv):
                fit = energy(dv)
                return [fit]

            def gradient(self, dv):
                grad = graden(dv)
                return grad

            def get_bounds(self):
                return ([-100]*dim_0, [100]*dim_0)

            def get_name(self):
                return "Poisson problem"

        prb = pg.problem(PoissonProblem())
        algo = pg.algorithm(pg.nlopt(solver="tnewton"))
        algo.extract(pg.nlopt).ftol_abs = 1e-5
        algo.extract(pg.nlopt).ftol_rel = 1e-5
        pop = pg.population(prb)
        pop.push_back(u_0)
        print(algo)
        # algo.set_verbosity(1)
        pop = algo.evolve(pop)
        u = pop.champion_x
        toc = time.time()

    print("Elapsed time = ", toc-tic)
    assert np.allclose(u[bnodes], u_true[bnodes], atol=1e-2)
    assert np.allclose(u, u_true, atol=1e-2)


if __name__ == '__main__':
    assert len(sys.argv) > 1
    optimizer = sys.argv[1]
    platform = sys.argv[2]
    bench_poisson(optimizer=optimizer, platform=platform)
