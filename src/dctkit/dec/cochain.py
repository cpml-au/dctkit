import dctkit as dt

from dctkit.mesh import simplex as spx
from dctkit.math import spmv
import numpy.typing as npt
from jax import Array
import jax.numpy as jnp
from typeguard import check_type


class Cochain():
    """Cochain class.

    Args:
        dim: dimension of the complex where the cochain is defined.
        is_primal: True if the cochain is primal, False otherwise.
        complex: the simplicial complex where the cochain is defined.
        coeffs: array of the coefficients of the cochain.
    """

    def __init__(self, dim: int, is_primal: bool, complex: spx.SimplicialComplex,
                 coeffs: npt.NDArray | Array):
        self.dim = dim
        self.complex = complex
        self.is_primal = is_primal
        check_type(coeffs, npt.NDArray | Array)
        self.coeffs = coeffs


class CochainP(Cochain):
    """Class for primal cochains."""

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs):
        super().__init__(dim, True, complex, coeffs)


class CochainD(Cochain):
    """Class for dual cochains."""

    def __init__(self, dim: int, complex: spx.SimplicialComplex, coeffs):
        super().__init__(dim, False, complex, coeffs)


class CochainP0(CochainP):
    """Class for primal 0-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(0, complex, coeffs)


class CochainP1(CochainP):
    """Class for primal 1-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(1, complex, coeffs)


class CochainP2(CochainP):
    """Class for primal 2-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(2, complex, coeffs)


class CochainD0(CochainD):
    """Class for dual 0-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(0, complex, coeffs)


class CochainD1(CochainD):
    """Class for dual 1-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(1, complex, coeffs)


class CochainD2(CochainD):
    """Class for dual 2-cochains."""

    def __init__(self, complex: spx.SimplicialComplex, coeffs):
        super().__init__(2, complex, coeffs)


def add(c1: Cochain, c2: Cochain) -> Cochain:
    """Adds two cochains.

    Args:
        c1: a cohcain.
        c2: a cochain.
    Returns:
        c1 + c2
    """
    c = Cochain(c1.dim, c1.is_primal, c1.complex, c1.coeffs + c2.coeffs)
    return c


def sub(c1: Cochain, c2: Cochain) -> Cochain:
    """Subtracts two cochains.

    Args:
        c1: a cochain.
        c2: a cochain.
    Returns:
        c_1 - c_2
    """
    c = Cochain(c1.dim, c1.is_primal, c1.complex, c1.coeffs - c2.coeffs)
    return c


def scalar_mul(c: Cochain, k: float) -> Cochain:
    """Multiplies a cochain by a scalar.

    Args:
        c: a cochain.
        k: a scalar.
    Returns:
        cochain with coefficients equal to k*(c.coeffs).
    """
    C = Cochain(c.dim, c.is_primal, c.complex, k*c.coeffs)
    return C


def cochain_mul(c1: Cochain, c2: Cochain) -> Cochain:
    """Multiplies two cochain component-wise

    Args:
        c1: a cochain.
        c2: a cochain.
    Returns:
        cochain with coefficients = c1*c2.
    """

    assert (c1.is_primal == c2.is_primal)
    return Cochain(c1.dim, c1.is_primal, c1.complex, c1.coeffs*c2.coeffs)


def identity(c: Cochain) -> Cochain:
    """Implements the identity operator.

    Args:
        c: a cochain.
    Returns:
        the same cochain.
    """
    return c


def sin(c: Cochain) -> Cochain:
    """Computes the sin of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to sin(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.sin(c.coeffs))
    return C


def arcsin(c: Cochain) -> Cochain:
    """Computes the arcsin of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to arcsin(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.arcsin(c.coeffs))
    return C


def cos(c: Cochain) -> Cochain:
    """Computes the cos of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to cos(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.cos(c.coeffs))
    return C


def arccos(c: Cochain) -> Cochain:
    """Computes the arcsin of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to arccos(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.arccos(c.coeffs))
    return C


def exp(c: Cochain) -> Cochain:
    """Compute the exp of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to exp(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.exp(c.coeffs))
    return C


def log(c: Cochain) -> Cochain:
    """Computes the natural log of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to log(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.log(c.coeffs))
    return C


def sqrt(c: Cochain) -> Cochain:
    """Compute the sqrt of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients equal to sqrt(c.coeffs).

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.sqrt(c.coeffs))
    return C


def square(c: Cochain) -> Cochain:
    """Computes the square of a cochain.

    Args:
        c: a cochain.
    Returns:
        cochain with coefficients squared.

    """
    C = Cochain(c.dim, c.is_primal, c.complex, dt.backend.square(c.coeffs))
    return C


def coboundary(c: Cochain) -> Cochain:
    """Implements the coboundary operator.

    Args:
        c: a cochain.
    Returns:
        the cochain obtained by taking the coboundary of c.
    """
    # initialize dc
    dc = Cochain(dim=c.dim + 1, is_primal=c.is_primal, complex=c.complex,
                 coeffs=jnp.empty_like(c.coeffs, dtype=dt.float_dtype))

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
                     is_primal=not c.is_primal, complex=c.complex,
                     coeffs=jnp.empty_like(c.coeffs, dtype=dt.float_dtype))

    if c.is_primal:
        star_c.coeffs = c.complex.hodge_star[c.dim]*c.coeffs
    else:
        # NOTE: this step only works with well-centered meshes!
        assert c.complex.is_well_centered
        star_c.coeffs = c.complex.hodge_star_inverse[c.complex.dim - c.dim]*c.coeffs

    return star_c


def inner_product(c1: Cochain, c2: Cochain) -> Array:
    """Computes the inner product between two cochains.

    Args:
        c1: a cochain.
        c2: a cochain.
    Returns:
        inner product between c1 and c2.
    """
    star_c_2 = star(c2)
    n = c1.complex.dim

    # dimension of the complexes must agree
    assert (n == c2.complex.dim)

    inner_product = dt.backend.dot(c1.coeffs, star_c_2.coeffs)
    # NOTE: not sure whether we should keep both Jax and numpy as backends and allow for
    # different return types
    check_type(inner_product, npt.NDArray | Array)
    return inner_product


def codifferential(c: Cochain) -> Cochain:
    """Implements the discrete codifferential.

    Args:
        c: a cochain.
    Returns:
        the discrete codifferential of c.
    """
    k = c.dim
    n = c.complex.dim
    cob = coboundary(star(c))
    if c.is_primal:
        return Cochain(k-1, c.is_primal, c.complex, (-1)**(n*(k-1)+1)*star(cob).coeffs)
    return Cochain(k-1, c.is_primal, c.complex, (-1)**(k*(n+1-k))*star(cob).coeffs)


def laplacian(c: Cochain) -> Cochain:
    """Implements the discrete Laplace-de Rham (or Laplace-Beltrami) operator.
    (https://en.wikipedia.org/wiki/Laplace%E2%80%93Beltrami_operator)

    Args:
        c: a cochain.
    Returns:
        a cochain.
    """
    if c.dim == 0:
        laplacian = codifferential(coboundary(c))
    else:
        laplacian = add(codifferential(coboundary(c)), coboundary(codifferential(c)))
    return laplacian
