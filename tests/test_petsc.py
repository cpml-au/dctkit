import numpy as np
from dctkit.mesh import util
import dctkit as dt
from dctkit.dec import cochain as C
from dctkit.math.opt import optctrl as oc


def test_petsc(setup_test):
    lc = 0.008

    mesh, _ = util.generate_square_mesh(lc)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    bnodes = mesh.cell_sets_dict["boundary"]["line"]
    node_coord = S.node_coords

    # NOTE: exact solution of Delta u + f = 0
    u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1] ** 2, dtype=dt.float_dtype)
    b_values = u_true[bnodes]

    boundary_values = (np.array(bnodes, dtype=dt.int_dtype), b_values)

    num_nodes = S.num_nodes
    f_vec = -4.*np.ones(num_nodes, dtype=dt.float_dtype)

    u_0 = np.zeros(num_nodes, dtype=dt.float_dtype)

    gamma = 1000.

    def energy_poisson(x, f, boundary_values, gamma):
        pos, value = boundary_values
        f = C.Cochain(0, True, S, f)
        u = C.Cochain(0, True, S, x)
        du = C.coboundary(u)
        norm_grad = 1/2.*C.inner(du, du)
        bound_term = -C.inner(u, f)
        penalty = 0.5*gamma*dt.backend.sum((x[pos] - value)**2)
        energy = norm_grad + bound_term + penalty
        return energy

    prb = oc.OptimizationProblem(
        num_nodes, num_nodes, energy_poisson, solver_lib="petsc")

    kargs = {"f": f_vec, "boundary_values": boundary_values, "gamma": gamma}
    prb.set_obj_args(kargs)

    u = prb.solve(u_0)

    assert np.allclose(u, u_true, atol=1e-2)
