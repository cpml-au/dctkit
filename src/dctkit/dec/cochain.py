import numpy as np

from dctkit.mesh import simplex as spx
from dctkit.math import spmv


class Cochain():
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

    def __init__(self, dim: int, is_primal: bool, complex: spx.SimplicialComplex,
                 coeffs=None):
        self.dim = dim
        self.complex = complex
        self.is_primal = is_primal
        self.coeffs = coeffs


def coboundary(c):
    """Implements the coboundary operator.

    Args:
        c (Cochain): a cochain.
    Returns:
        dc (Cochain): the cochain obtained taking the coboundary of c.
    """
    # initialize dc
    dc = Cochain(dim=c.dim + 1, is_primal=c.is_primal, complex=c.complex)

    # get the appropriate (primal) boundary matrix
    t = c.complex.boundary[c.dim + 1]

    # apply coboundary matrix (transpose of the primal boundary matrix) to the
    # array of coefficients of the cochain.
    if c.is_primal:
        dc.coeffs = spmv.spmv_coo(t, c.coeffs, transpose=True)
    else:
        # FIXME: transpose is not enough to compute the dual coboundary op
        dc.coeffs = spmv.spmv_coo(t, c.coeffs, transpose=False)
    return dc


def star(c):
    """Implements the hodge star operator.

    Args:
        c (Cochain): a primal cochain.
    Returns:
        star_c (Cochain): the dual cochain *c obtained applying the hodge star operator.
    """
    star_c = Cochain(dim=c.complex.dim - c.dim, is_primal=False, complex=c.complex)
    star_c.coeffs = c.complex.hodge_star[c.dim]*c.coeffs
    return star_c


def inner_product(c_1, c_2):
    """Implements the inner product between two primal cochains.

    Args:
        c_1 (Cochain): a primal cochain.
        c_2 (Cochain): a primal cochain.
    Returns:
        inner_product (float): inner product between c_1 and c_2
    """
    star_c_2 = star(c_2)
    n = c_1.complex.dim

    # dimension of the complexes must agree
    assert (n == c_2.complex.dim)

    inner_product = 1/n*np.sum(c_1.coeffs*star_c_2.coeffs)
    return inner_product
