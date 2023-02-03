import numpy as np

from dctkit.mesh import simplex as spx
from dctkit.math import spmv
import jax.numpy as jnp


class Cochain():
    """Cochain associated to a simplicial complex.

    Args:
        dim (int): dimension of the chains in which the cochain is defined.
        is_primal (bool): boolean which is True if the cochain is primal and it
            is False if it is dual.
        node_tags (np.array): np.array matrix of node tags.
        vec (np.array): vectorial representation of the cochain.
    Attributes:
        dim (int): dimension of the chains in which the cochain is defined.
        is_primal (bool): boolean which is True if the cochain is primal and it
            is False if it is dual.
        node_tags (np.array): inherited from the class simplicial_complex.
        vec (np.array): vectorial representation of the cochain.
    """

    def __init__(self, dim: int, is_primal: bool, complex: spx.SimplicialComplex,
                 coeffs=None, type="numpy"):
        self.dim = dim
        self.complex = complex
        self.is_primal = is_primal
        self.coeffs = coeffs
        self.type = type


class CochainP(Cochain):
    "Class for primal cochains"

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs=None,
                 type="numpy"):
        super().__init__(dim, True, complex, coeffs, type)


class CochainD(Cochain):
    "Class for dual cochains"

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs=None,
                 type="numpy"):
        super().__init__(dim, False, complex, coeffs, type)


class CochainP0(CochainP):
    """Class for primal 0-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(0, complex, coeffs, type)


class CochainP1(CochainP):
    """Class for primal 1-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(1, complex, coeffs, type)


class CochainP2(CochainP):
    """Class for primal 2-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(2, complex, coeffs, type)


class CochainD0(CochainD):
    """Class for dual 0-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(0, complex, coeffs, type)


class CochainD1(CochainD):
    """Class for dual 1-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(1, complex, coeffs, type)


class CochainD2(CochainD):
    """Class for dual 2-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None, type="numpy"):
        super().__init__(2, complex, coeffs, type)


def add(c_1, c_2):
    """Implements the sum operator for two cochains of the same dimension.

    Args:
        c_1 (Cochain): a cochain
        c_2 (Cochain): another cochain with the same dimension of c_1
    Returns:
        Cochain: c_1 + c_2
    """
    c = Cochain(c_1.dim, c_1.is_primal, c_1.complex, c_1.coeffs + c_2.coeffs, c_1.type)
    return c


def scalar_mul(c, k):
    """Implements the scalar multiplication operator.

    Args:
        c (Cochain): a cochain.
        k (float): a float.
    Returns:
        Cochain: cochain with coefficients equal to k*(c.coeffs)

    """
    C = Cochain(c.dim, c.is_primal, c.complex, k*c.coeffs, c.type)
    return C


def identity(c):
    """Implements the identity operator.

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: the same cochain.

    """
    return c


def coboundary(c):
    """Implements the coboundary operator.

    Args:
        c (Cochain): a cochain.
    Returns:
        dc (Cochain): the cochain obtained taking the coboundary of c.
    """
    # initialize dc
    dc = Cochain(dim=c.dim + 1, is_primal=c.is_primal, complex=c.complex, type=c.type)

    # apply coboundary matrix (transpose of the primal boundary matrix) to the
    # array of coefficients of the cochain.
    if c.is_primal:
        # get the appropriate (primal) boundary matrix
        t = c.complex.boundary[c.dim + 1]
        dc.coeffs = spmv.spmv_coo(t, c.coeffs, transpose=True,
                                  shape=c.complex.S[c.dim+1].shape[0])
    else:
        t = c.complex.boundary[c.complex.dim - c.dim]
        dc.coeffs = spmv.spmv_coo(t, c.coeffs, transpose=False,
                                  shape=c.complex.S[c.complex.dim-c.dim-1].shape[0])
    return dc


def star(c):
    """Implements the hodge star operator.

    Args:
        c (Cochain): a primal cochain.
    Returns:
        star_c (Cochain): the dual cochain *c obtained applying the hodge star operator.
    """
    star_c = Cochain(dim=c.complex.dim - c.dim,
                     is_primal=not c.is_primal, complex=c.complex)
    if c.is_primal:
        star_c.coeffs = c.complex.hodge_star[c.dim]*c.coeffs
    else:
        # NOTE: this step only works with well-centered meshes!
        star_c.coeffs = c.complex.hodge_star_inverse[c.complex.dim - c.dim]*c.coeffs
    return star_c


def inner_product(c_1, c_2):
    """Implements the inner product between two primal cochains.

    Args:
        c_1 (Cochain): a primal cochain.
        c_2 (Cochain): a primal cochain.
    Returns:
        float: inner product between c_1 and c_2.
    """
    star_c_2 = star(c_2)
    n = c_1.complex.dim

    # dimension of the complexes must agree
    assert (n == c_2.complex.dim)

    assert (c_1.type == c_2.type)

    if c_1.type == "numpy":
        inner_product = np.dot(c_1.coeffs, star_c_2.coeffs)
    elif c_1.type == "jax":
        inner_product = jnp.dot(c_1.coeffs, star_c_2.coeffs)
    return inner_product


def codifferential(c):
    """Implements the discrete codifferential.

    Args:
        c: a cochain.
    Returns:
        (Cochain): the discrete codifferential of c.
    """
    k = c.dim
    n = c.complex.dim
    cob = coboundary(star(c))
    d_star_c = Cochain(k-1, c.is_primal, c.complex, (-1)**(n*(k-1)+1)*star(cob).coeffs)
    # NOTE: since for us the dual coboundary is just the transpose, we have to adjust
    # the sign multiplying d_star_c with (-1)^k
    d_star_c.coeffs *= (-1)**k
    return d_star_c
