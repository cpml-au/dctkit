from dctkit.math.opt import optctrl
import jax
import jax.numpy as jnp
import numpy as np
import dctkit as dt
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
import gmsh
import matplotlib.tri as tri
from dctkit import config, FloatDtype, IntDtype, Backend, Platform

config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)


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


def test_optimal_control_toy():
    target = np.array([2., 1.], dtype=np.float32)

    def statefun(x: np.array, y: np.array) -> float:
        """Discrete functional associated to the Minimum problem that determines the
        state.

        Args:
            x: state vector.
            y: paramters (controls).
        Returns:
            state vector that minimizes the functional.
        """
        return jnp.sum(jnp.square(jnp.square(x)-y))

    def objfun(x: np.array, y: np.array) -> float:
        """Objective function. Problem: choose y such that the state x(y) minimizes the
        distance wrt to the target.

        Args:
            x: state vector.
            y: paramters (controls).
        """
        return jnp.sum(jnp.square(x-target))

    # initial guesses
    x0 = np.ones(2)
    y0 = np.zeros(2)

    prb = optctrl.OptimalControlProblem(objfun=objfun, state_en=statefun, state_dim=2)
    x, y, fval = prb.run(x0, y0, tol=1e-6)
    print(x, y, fval)

    assert np.allclose(y, target**2, atol=1e-4)
    assert np.allclose(x, target, atol=1e-4)


def test_optimal_control_poisson():

    if jax.config.read("jax_enable_x64"):
        assert dt.float_dtype == "float64"

    np.random.seed(42)

    lc = 0.5

    _, _, S_2, node_coord = util.generate_square_mesh(lc)

    S, bnodes, _ = get_complex(S_2, node_coord)

    k = 1.

    # NOTE: exact solution of Delta u + f = 0
    u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1]
                      ** 2, dtype=dt.float_dtype)
    b_values = u_true[bnodes]

    boundary_values = (np.array(bnodes, dtype=dt.int_dtype), b_values)

    dim_0 = S.num_nodes
    f_true = -4.*np.ones(1, dtype=dt.float_dtype)

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0).astype(dt.float_dtype)
    u_0 = np.array(u_0, dtype=dt.float_dtype)

    gamma = 100000.

    def energy_poisson(x: np.array, f: np.array) -> float:
        pos, value = boundary_values
        f = C.Cochain(0, True, S, f*np.ones(dim_0, dtype=dt.float_dtype))
        u = C.Cochain(0, True, S, x)
        du = C.coboundary(u)
        norm_grad = k/2.*C.inner_product(du, du)
        bound_term = -C.inner_product(u, f)
        penalty = 0.5*gamma*dt.backend.sum((x[pos] - value)**2)
        energy = norm_grad + bound_term + penalty
        return energy

    def obj_fun(x: np.array, f: np.array) -> float:
        return jnp.sum(jnp.square(x-u_true))

    # initial guesses
    f0 = -1.*np.ones(1, dtype=dt.float_dtype)

    prb = optctrl.OptimalControlProblem(
        objfun=obj_fun, state_en=energy_poisson, state_dim=dim_0)
    x, f, fval = prb.run(u_0, f0, tol=1e-2)
    print("dim = ", dim_0)
    print("u = ", x)
    print("u_true = ", u_true)
    print("f = ", f)
    print("fval = ", fval)

    assert np.allclose(x, u_true, atol=1e-4)
    assert np.allclose(f, f_true, atol=1e-4)


if __name__ == "__main__":
    test_optimal_control_poisson()
    test_optimal_control_toy()
