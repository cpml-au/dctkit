import dctkit as dt

from dctkit.mesh import simplex as spx
from dctkit.math import spmv


class Cochain():
    """Cochain class.

    Args:
        dim (int): dimension of the complex where the cochain is defined.
        is_primal (bool): True if the cochain is primal, False otherwise.
        complex: a SimplicialComplex object.
        coeffs: array of the coefficients of the cochain.
    """

    def __init__(self, dim: int, is_primal: bool, complex: spx.SimplicialComplex,
                 coeffs=None):
        self.dim = dim
        self.complex = complex
        self.is_primal = is_primal
        self.coeffs = coeffs


class CochainP(Cochain):
    "Class for primal cochains"

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(dim, True, complex, coeffs)


class CochainD(Cochain):
    "Class for dual cochains"

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(dim, False, complex, coeffs)


class CochainP0(CochainP):
    """Class for primal 0-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(0, complex, coeffs)


class CochainP1(CochainP):
    """Class for primal 1-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(1, complex, coeffs, type)


class CochainP2(CochainP):
    """Class for primal 2-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(2, complex, coeffs, type)


class CochainD0(CochainD):
    """Class for dual 0-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(0, complex, coeffs, type)


class CochainD1(CochainD):
    """Class for dual 1-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(1, complex, coeffs, type)


class CochainD2(CochainD):
    """Class for dual 2-cochains"""

    def __init__(self, complex: spx.SimplicialComplex, coeffs=None):
        super().__init__(2, complex, coeffs, type)


def add(c_1: Cochain, c_2: Cochain) -> Cochain:
    """Adds two p-cochains.

    Args:
        c_1 (Cochain): a p-cochain.
        c_2 (Cochain): a p-cochain.
    Returns:
        Cochain: c_1 + c_2
    """
    c = Cochain(c_1.dim, c_1.is_primal, c_1.complex, c_1.coeffs + c_2.coeffs)
    return c


def sub(c_1: Cochain, c_2: Cochain) -> Cochain:
    """Subtracts two p-cochains.

    Args:
        c_1 (Cochain): a p-cochain.
        c_2 (Cochain): a p-cochain.
    Returns:
        Cochain: c_1 - c_2
    """
    c = Cochain(c_1.dim, c_1.is_primal, c_1.complex, c_1.coeffs - c_2.coeffs)
    return c


def scalar_mul(c: Cochain, k: float) -> Cochain:
    """Multiplies a cochain by a scalar.

    Args:
        c (Cochain): a cochain.
        k (float): a scalar.
    Returns:
        Cochain: cochain with coefficients equal to k*(c.coeffs).
    """
    C = Cochain(c.dim, c.is_primal, c.complex, k*c.coeffs)
    return C


def identity(c: Cochain) -> Cochain:
    """Implements the identity operator.

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: the same cochain.
    """
    return c


def sin(c: Cochain) -> Cochain:
    """Compute the sin of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to sin(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.sin(c.coeffs))
    return C


def arcsin(c: Cochain) -> Cochain:
    """Compute the arcsin of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to arcsin(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.arcsin(c.coeffs))
    return C


def cos(c: Cochain) -> Cochain:
    """Compute the cos of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to cos(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.cos(c.coeffs))
    return C


def arccos(c: Cochain) -> Cochain:
    """Compute the arcsin of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to arccos(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.arccos(c.coeffs))
    return C


def exp(c: Cochain) -> Cochain:
    """Compute the exp of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to exp(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.exp(c.coeffs))
    return C


def log(c: Cochain) -> Cochain:
    """Compute the natural log of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to log(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.log(c.coeffs))
    return C


def sqrt(c: Cochain) -> Cochain:
    """Compute the sqrt of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to (c.coeffs)^(1/2).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.sqrt(c.coeffs))
    return C


def square(c: Cochain) -> Cochain:
    """Compute the square of a cochain

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: cochain with coefficients equal to (c.coeffs)^2.

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.square(c.coeffs))
    return C


def coboundary(c: Cochain) -> Cochain:
    """Implements the coboundary operator.

    Args:
        c (Cochain): a cochain.
    Returns:
        Cochain: the cochain obtained by taking the coboundary of c.
    """
    # initialize dc
    dc = Cochain(dim=c.dim + 1, is_primal=c.is_primal, complex=c.complex)

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


def star(c: Cochain) -> Cochain:
    """Implements the diagonal Hodge star operator (see Grinspun et al.).

    Args:
        c: a cochain.
    Returns:
        the dual cochain obtained applying the Hodge star operator.
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


def inner_product(c_1: CochainP, c_2: CochainP) -> float:
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

    inner_product = dt.backend.dot(c_1.coeffs, star_c_2.coeffs)
    return inner_product


def codifferential(c: Cochain) -> Cochain:
    """Implements the discrete codifferential.

    Args:
        c: a cochain.
    Returns:
        Cochain: the discrete codifferential of c.
    """
    k = c.dim
    n = c.complex.dim
    cob = coboundary(star(c))
    d_star_c = Cochain(k-1, c.is_primal, c.complex, (-1)
                       ** (n*(k-1)+1)*star(cob).coeffs)
    return d_star_c


def laplacian(c: Cochain) -> Cochain:
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
