import numpy as np
from scipy.optimize import minimize
from dctkit.mesh import simplex, util
from dctkit.apps import poisson as p
from dctkit.dec import cochain as C
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from icecream import ic

import gmsh


cwd = os.path.dirname(simplex.__file__)


def test_poisson():
    # tested with test1.msh, test2.msh and test3.msh
    filename = "test3.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)
    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    triang = tri.Triangulation(node_coord[:, 0], node_coord[:, 1])

    plt.triplot(triang)
    plt.show()

    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_2, node_coord)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # TODO: initialize diffusivity
    k = 1.

    # exact solution
    u_true = node_coord[:, 0]**2 + node_coord[:, 1]**2
    b_values = u_true[bnodes]

    plt.tricontourf(triang, u_true, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()

    # TODO: initialize boundary_values
    boundary_values = (np.array(bnodes), b_values)
    # TODO: initialize external sources
    dim_0 = S.num_nodes
    f = C.Cochain(0, True, S, 4.*np.ones(dim_0))
    star_f = C.star(f)

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0)

    # penalty factor on boundary conditions
    gamma = 10.
    args = (star_f.coeffs, S, k, boundary_values, gamma)
    u = minimize(fun=p.obj_poisson, x0=u_0, args=args, method='BFGS',
                 jac=p.grad_poisson, options={'disp': 1})

    assert np.allclose(u.x[bnodes], u_true[bnodes], atol=1e-6)
    assert np.allclose(u.x, u_true, atol=1e-6)

    plt.tricontourf(triang, u.x, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()


def test_energy_poisson():
    filename = "test2.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)
    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    triang = tri.Triangulation(node_coord[:, 0], node_coord[:, 1])

    plt.triplot(triang)
    plt.show()

    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_2, node_coord)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # TODO: initialize diffusivity
    k = 1.

    dim_0 = S.num_nodes

    # exact solution
    u_true = node_coord[:, 0]**2 + node_coord[:, 1]**2
    b_values = u_true[bnodes]

    # TODO: initialize boundary_values
    boundary_values = (np.array(bnodes), b_values)
    # TODO: initialize external sources
    f = 4.*np.ones(dim_0)

    plt.tricontourf(triang, u_true, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()

    # initial guess
    u_0 = 0.01*np.random.rand(dim_0)

    # penalty factor on boundary conditions
    gamma = 100000.

    args = (f, S, k, boundary_values, gamma)
    u = minimize(fun=p.energy_poisson, x0=u_0, args=args, method='BFGS',
                 jac=p.grad_energy_poisson, options={'disp': 1})

    plt.tricontourf(triang, u.x, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()

    ic(np.linalg.norm(u.x-u_true))
    ic(np.linalg.norm(u.x[bnodes]-u_true[bnodes]))
    assert np.allclose(u.x[bnodes], u_true[bnodes], atol=1e-6)
    assert np.allclose(u.x, u_true, atol=1e-6)


if __name__ == '__main__':
    # test_poisson()
    test_energy_poisson()
