import numpy as np

from dctkit.mesh import simplex as spx
from dctkit.math import spmv
import jax.numpy as jnp


class Cochain():
    """Cochain class.

    Args:
        dim (int): dimension of the complex where the cochain is defined.
        is_primal (bool): True if the cochain is primal, False otherwise.
        complex: a SimplicialComplex object.
        coeffs: array of the coefficients of the cochain.
        type (str): either "numpy" of "jax" according to the type (numpy array or jax
            array) of the coefficients.
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
    """Adds two p-cochains.

    Args:
        c_1 (Cochain): a p-cochain.
        c_2 (Cochain): a p-cochain.
    Returns:
        Cochain: c_1 + c_2
    """
    assert (c_1.type == c_2.type)
    c = Cochain(c_1.dim, c_1.is_primal, c_1.complex, c_1.coeffs + c_2.coeffs, c_1.type)
    return c


def sub(c_1, c_2):
    """Subtracts two p-cochains.

    Args:
        c_1 (Cochain): a p-cochain.
        c_2 (Cochain): a p-cochain.
    Returns:
        Cochain: c_1 - c_2
    """
    assert (c_1.type == c_2.type)
    c = Cochain(c_1.dim, c_1.is_primal, c_1.complex, c_1.coeffs - c_2.coeffs, c_1.type)
    return c


def scalar_mul(c, k):
    """Multiplies a cochain by a scalar.

    Args:
        c (Cochain): a cochain.
        k (float): a scalar.
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
        Cochain: the cochain obtained by taking the coboundary of c.
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
        dc.coeffs = spmv.spmv_coo(t, c.coeffs,
                                  transpose=False,
                                  shape=c.complex.S[c.complex.dim-c.dim-1].shape[0])
        dc.coeffs *= (-1)**(c.complex.dim - c.dim)
    return dc


def star(c):
    """Implements the hodge star operator.

    Args:
        c (Cochain): a primal cochain.
    Returns:
        Cochain: the dual cochain obtained applying the hodge star operator.
    """
    star_c = Cochain(dim=c.complex.dim - c.dim,
                     is_primal=not c.is_primal, complex=c.complex)
    if c.is_primal:
        star_c.coeffs = c.complex.hodge_star[c.dim]*c.coeffs
    else:
        # NOTE: this step only works with well-centered meshes!
        assert c.complex.is_well_centered
        star_c.coeffs = c.complex.hodge_star_inverse[c.complex.dim - c.dim]*c.coeffs
    return star_c


def inner_product(c_1, c_2):
    """Computes the inner product between two primal cochains.

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
        Cochain: the discrete codifferential of c.
    """
    k = c.dim
    n = c.complex.dim
    cob = coboundary(star(c))
    d_star_c = Cochain(k-1, c.is_primal, c.complex, (-1) **
                       (n*(k-1)+1)*star(cob).coeffs, type=c.type)
    return d_star_c


def laplacian(c):
    """Implements the discrete laplacian operator.

    Args:
        c: a k-cochain.
    Returns:
        Cochain: a k-cochain obtained taking the discrete laplacian of c.
    """
    if c.dim == 0:
        laplacian = codifferential(coboundary(c))
    else:
        laplacian = codifferential(coboundary(c)) + coboundary(codifferential(c))
    return laplacian
