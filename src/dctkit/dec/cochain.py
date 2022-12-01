from dctkit.mesh import simplex
from dctkit.math import spmv


class Cochain(simplex.SimplicialComplex):
    """Cochain associated to a simplicial complex

    Args:
        dim (int): dimension of the chains in which the cochain is defined.
        is_primal (bool): boolean which is True if the cochain is primal
                          and it is False if it is dual.
        node_tags (np.array): np.array matrix of node tags.
        vec (np.array): vectorial representation of the cochain
    Attributes:
        dim (int): dimension of the chains in which the cochain is defined.
        is_primal (bool): boolean which is True if the cochain is primal
                          and it is False if it is dual.
        node_tags (np.array): inherited from the class simplicial_complex
        vec (np.array): vectorial representation of the cochain.
    """

    def __init__(self, dim: int, is_primal: bool, node_tags, vec=None):
        super().__init__(node_tags)
        self.dim = dim
        self.is_primal = is_primal
        self.vec = vec


def coboundary_operator(c):
    """Implements the coboundary operator

    Args:
        c (Cochain): a cochain
    Returns:
        dc (Cochain): the cochain obtained taking the coboundary of c
    """
    # initialize dc
    dc = Cochain(dim=c.dim + 1, is_primal=c.is_primal, node_tags=c.node_tags)

    # construct boundary matrix
    t = c.get_boundary_operators()[c.dim]

    # check if transposition is needed
    # and compute matrix-vector product
    if c.is_primal:
        dc.vec = spmv.spmv_coo(t, c.vec, transpose=True)
    else:
        dc.vec = spmv.spmv_coo(t, c.vec, transpose=False)
    return dc
