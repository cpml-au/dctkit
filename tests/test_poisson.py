import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
from examples import poisson
import os

cwd = os.path.dirname(simplex.__file__)
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
k = 0.1
# TODO: initialize boundary_values
boundary_values = (np.array([0, 1, 2, 3]), np.ones(4))


def Poisson_vec_operator(v):
    # S is a predefined SimplicialComplex object
    c = C.Cochain(0, True, S, v)
    # k and boundary_values are predefined
    p = poisson.Poisson(c, k, boundary_values)
    w = p.coeffs
    print(w)
    return w


def test_poisson():
    # TODO: initialize external sources
    zero_coch_dim = node_coord.shape[0]
    f_coeffs = np.zeros(zero_coch_dim)
    Poisson = LinearOperator((zero_coch_dim, zero_coch_dim),
                             matvec=Poisson_vec_operator)
    u_coeffs = cg(Poisson, f_coeffs)
    print(u_coeffs)


if __name__ == '__main__':
    test_poisson()
