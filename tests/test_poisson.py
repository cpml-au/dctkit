import numpy as np
from scipy.optimize import fmin_cg
from dctkit.mesh import simplex, util
from examples import poisson as p
import os

cwd = os.path.dirname(simplex.__file__)


def test_poisson():
    filename = "test1.msh"
    full_path = os.path.join(cwd, filename)
    numNodes, numElements, S_2, node_coord = util.read_mesh(full_path)

    print(f"The number of nodes in the mesh is {numNodes}")
    print(f"The number of faces in the mesh is {numElements}")
    print(f"The vectorization of the face matrix is \n {S_2}")

    # initialize simplicial complex
    S = simplex.SimplicialComplex(S_2, node_coord)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    # TODO: initialize diffusivity
    k = 1
    # TODO: initialize boundary_values
    boundary_values = (np.array([0, 1, 2, 3]), np.ones(4))
    # TODO: initialize external sources
    dim_0 = node_coord.shape[0]
    f = np.zeros(dim_0)
    u_0 = np.random.randint(100, size=(5, 1))
    gamma = 10000
    args = (f, S, k, boundary_values, gamma)
    u = fmin_cg(p.obj_poisson, u_0, args=args, fprime=p.grad_poisson, disp=1,
                full_output=True)
    assert (u[1] < 10**-3)


if __name__ == '__main__':
    test_poisson()
