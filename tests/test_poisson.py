import numpy as np
from scipy.optimize import minimize
from dctkit.mesh import simplex, util
from dctkit.apps import poisson as p
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from icecream import ic
import gmsh


cwd = os.path.dirname(simplex.__file__)


def test_poisson():
    filename = "test3.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)
    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    ic(bnodes)

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
    # TODO: initialize boundary_values
    boundary_values = (np.array(bnodes), np.zeros(len(bnodes)))
    # TODO: initialize external sources
    dim_0 = S.num_nodes
    f = 1*np.ones(dim_0)
    u_0 = 0.01*np.random.rand(dim_0)
    # u_0 = np.zeros(dim_0)
    ic(u_0)

    gamma = 10.
    args = (f, S, k, boundary_values, gamma)
    u = minimize(fun=p.obj_poisson, x0=u_0, args=args, method='CG',
                 jac=p.grad_poisson, options={'disp': 1})

    ic(u.x)
    plt.tricontourf(triang, u.x, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    test_poisson()
