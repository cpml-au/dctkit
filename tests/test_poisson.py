import numpy as np
from scipy.optimize import minimize
from dctkit.mesh import simplex
from dctkit.apps import poisson as p
from dctkit.dec import cochain as C
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from icecream import ic

import gmsh


cwd = os.path.dirname(simplex.__file__)


def generate_mesh(lc):
    gmsh.model.add("t1")
    gmsh.model.geo.addPoint(1, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(0, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(3, 1, 4)
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [1, 2, 3, 4], 1)
    gmsh.model.mesh.generate(2)

    # Get nodes and corresponding coordinates
    nodeTags, coords, paramCoords = gmsh.model.mesh.getNodes()
    numNodes = len(nodeTags)
    # print("# nodes = ", numNodes)

    # Get 2D elements and associated node tags
    # NOTE: ONLY GET TRIANGLES
    elemTags, nodeTagsPerElem = gmsh.model.mesh.getElementsByType(2)

    # Decrease element IDs by 1 to have node indices starting from 0
    nodeTagsPerElem = np.array(nodeTagsPerElem) - 1
    nodeTagsPerElem = nodeTagsPerElem.reshape(len(nodeTagsPerElem) // 3, 3)
    # Get number of TRIANGLES
    numElements = len(elemTags)
    # print("# elements = ", numElements)

    # physicalGrps = gmsh.model.getPhysicalGroups()
    # print("physical groups: ", physicalGrps)

    # edgeNodesTags = gmsh.model.mesh.getElementEdgeNodes(2)
    # print("edge nodes tags: ", edgeNodesTags)

    # Position vectors of mesh points
    node_coords = coords.reshape(len(coords)//3, 3)
    return numNodes, numElements, nodeTagsPerElem, node_coords


def test_poisson(energy_bool):
    # tested with test1.msh, test2.msh and test3.msh
    # filename = "test1.msh"
    # full_path = os.path.join(cwd, filename)
    gmsh.initialize()
    history = []
    history_boundary = []
    final_energy = []
    lc = 1.0
    i = 2
    for j in range(i):
        _, _, S_2, node_coord = generate_mesh(lc)

        # numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

        bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
        bnodes -= 1

        triang = tri.Triangulation(node_coord[:, 0], node_coord[:, 1])

        plt.triplot(triang)
        plt.show()

        # initialize simplicial complex
        ic()
        S = simplex.SimplicialComplex(S_2, node_coord)
        ic()
        S.get_circumcenters()
        S.get_primal_volumes()
        S.get_dual_volumes()
        S.get_hodge_star()
        ic()
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
        f_vec = 4.*np.ones(dim_0)

        mask = np.ones(dim_0)
        mask[bnodes] = 0.

        if energy_bool:
            obj = p.energy_poisson
            grad = p.grad_energy_poisson
            gamma = 100000.
            args = (f_vec, S, k, boundary_values, gamma)
        else:
            obj = p.obj_poisson
            grad = p.grad_poisson
            f = C.Cochain(0, True, S, f_vec)
            star_f = C.star(f)
            # penalty factor on boundary conditions
            gamma = 10000.
            args = (star_f.coeffs, S, k, boundary_values, gamma, mask)

        # initial guess
        u_0 = 0.01*np.random.rand(dim_0)

        u = minimize(fun=obj, x0=u_0, args=args, method='BFGS',
                     jac=grad, options={'disp': 1})

        plt.tricontourf(triang, u.x, cmap='RdBu', levels=20)
        plt.triplot(triang, 'ko-')
        plt.colorbar()
        plt.show()

        ic(np.linalg.norm(u.x-u_true), np.linalg.norm(u.x[bnodes]-u_true[bnodes]))
        history.append(np.linalg.norm(u.x-u_true))
        history_boundary.append(np.linalg.norm(u.x[bnodes]-u_true[bnodes]))
        final_energy.append(u.fun)
        lc = lc/np.sqrt(2)

        # assert np.allclose(u.x[bnodes], u_true[bnodes], atol=1e-6)
        # assert np.allclose(u.x, u_true, atol=1e-6)

    plt.plot(range(i), history, label="Error")
    plt.plot(range(i), history_boundary, label="Boundary Error")
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(range(i), final_energy, label="Final Energy")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    test_poisson(True)
