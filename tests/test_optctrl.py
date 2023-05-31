from dctkit.math.opt import optctrl
import jax
import jax.numpy as jnp
import numpy as np
import dctkit as dt
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
import gmsh
import matplotlib.tri as tri
from dctkit import config, FloatDtype, Platform
import numpy.typing as npt
from jax import grad

config(FloatDtype.float32, Platform.cpu)


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


def test_optimal_control_toy(setup_test):
    # target state
    target = np.array([2., 1.], dtype=dt.float_dtype)

    state_dim = len(target)

    # total number of paramters (state + controls)
    nparams = 2*state_dim

    def statefun(x: npt.NDArray) -> jax.Array:
        """Discrete functional associated to the minimum problem that determines the
        state.

        Args:
            x: parameters array (state + controls).
        Returns:
            value of the energy of the system.
        """
        u = x[:state_dim]
        y = x[state_dim:]
        return grad(lambda u, y: jnp.sum(jnp.square(jnp.square(u)-y)))(u, y)

    def objfun(x: npt.NDArray) -> jax.Array:
        """Objective function. Problem: choose y such that the state x(y) minimizes the
        distance wrt to the target.

        Args:
            x: parameters array (state + controls).
        Returns:
            value of the objective function.
        """
        u = x[:state_dim]
        return jnp.sum(jnp.square(u-target))

    # initial guesses
    u0 = np.ones(2)
    y0 = np.zeros(2)
    x0 = np.concatenate((u0, y0))

    prb = optctrl.OptimalControlProblem(
        objfun=objfun, statefun=statefun, state_dim=state_dim, nparams=nparams)

    x = prb.run(x0=x0)
    u = x[:state_dim]
    y = x[state_dim:]

    assert np.allclose(y, target**2, atol=1e-4)
    assert np.allclose(u, target, atol=1e-4)


def test_optimal_control_poisson(setup_test):

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

    def energy_poisson(u: npt.NDArray, f: npt.NDArray) -> float:
        pos, value = boundary_values
        f_coch = C.CochainP0(S, f*np.ones(dim_0, dtype=dt.float_dtype))
        u_coch = C.CochainP0(S, u)
        du = C.coboundary(u_coch)
        norm_grad = k/2.*C.inner_product(du, du)
        bound_term = -C.inner_product(u_coch, f_coch)
        penalty = 0.5*gamma*dt.backend.sum((u[pos] - value)**2)
        energy = norm_grad + bound_term + penalty
        return energy

    def statefun(x: npt.NDArray) -> jax.Array:
        u = x[:dim_0]
        f = x[dim_0:]
        return grad(energy_poisson)(u, f)

    def objfun(x: npt.NDArray) -> jax.Array:
        u = x[:dim_0]
        return jnp.sum(jnp.square(u-u_true))

    # initial guesses
    f0 = -1.*np.ones(1, dtype=dt.float_dtype)

    x0 = np.concatenate((u_0, f0))

    prb = optctrl.OptimalControlProblem(
        objfun=objfun, statefun=statefun, state_dim=dim_0, nparams=dim_0 + len(f0))
    x = prb.run(x0=x0)
    u = x[:dim_0]
    f = x[dim_0:]
    print("u = ", u)
    print("u_true = ", u_true)
    print("f = ", f)

    assert np.allclose(u, u_true, atol=1e-4)
    assert np.allclose(f, f_true, atol=1e-4)
