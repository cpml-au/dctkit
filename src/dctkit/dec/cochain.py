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


str_init = """
def init(self, complex, coeffs):
    # if not is_primal_:
    self.dim = dim_
    self.is_primal = is_primal_
    self.complex = complex
    self.coeffs = coeffs
"""
attributes = {'category': (True, False), 'dim': (
    0, 1, 2, 3), 'rank': ("", "V", "T")}
categories = attributes['category']
dimensions = attributes['dim']
ranks = attributes['rank']
for is_primal_ in categories:
    for dim_ in dimensions:
        for rank_ in ranks:
            category_name = is_primal_*'P' + (not is_primal_)*'D'
            name = "Cochain" + category_name + str(dim_) + rank_

            exec(str_init.replace("dim_", f"{dim_}").replace(
                "is_primal_", f"{is_primal_}"))

            exec(name + " =type(name, (Cochain,), {'__init__': init})")


class CochainP(Cochain):
    """Class for primal cochains."""

    def __init__(self, dim: int, complex: spx.SimplicialComplex,
                 coeffs: npt.NDArray | Array):
        super().__init__(dim, True, complex, coeffs)


class CochainD(Cochain):
    """Class for dual cochains."""

    def __init__(self, dim: int, complex: spx.SimplicialComplex,
                 coeffs: npt.NDArray | Array):
        super().__init__(dim, False, complex, coeffs)


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
        cbnd_coo = c.complex.boundary[c.dim + 1]
        dc.coeffs = spmv.spmm(cbnd_coo, c.coeffs, transpose=True,
                              shape=c.complex.S[c.dim+1].shape[0])
    else:
        # FIXME: change sign of the boundary before applying it?
        bnd_coo = c.complex.boundary[c.complex.dim - c.dim]
        dc.coeffs = spmv.spmm(bnd_coo, c.coeffs,
                              transpose=False,
                              shape=c.complex.S[c.complex.dim-c.dim-1].shape[0])
        dc.coeffs *= (-1)**(c.complex.dim - c.dim)
    return dc


def coboundary_closure(c: CochainP) -> CochainD:
    """Implements the operator that complements the coboundary on the boundary
    of dual (n-1)-simplices, where n is the dimension of the complex.

    Args:
        c: a primal (n-1)-cochain
    Returns:
        the coboundary closure of c, resulting in a dual n-cochain with non-zero
        coefficients in the "uncompleted" cells.

    """
    n = c.complex.dim
    num_tets = c.complex.S[n].shape[0]
    num_dual_faces = c.complex.S[n-1].shape[0]

    # to extract only the boundary components with the right orientation
    # we construct a dual n-2 cochain and we take the (true) coboundary.
    # In this way the obtain a cochain such that an entry is 0 if it's in
    # the interior of the complex and ±1 if it's in the boundary
    ones = CochainD(dim=n-2, complex=c.complex, coeffs=jnp.ones(num_tets))
    diagonal_elems = coboundary(ones).coeffs
    diagonal_matrix_rows = jnp.arange(num_dual_faces)
    diagonal_matrix_cols = diagonal_matrix_rows
    diagonal_matrix_COO = [diagonal_matrix_rows, diagonal_matrix_cols, diagonal_elems]

    # build the absolute value of the (n-1)-coboundary
    abs_dual_coboundary_faces = c.complex.boundary[n-1].copy()
    # same of doing abs(dual_coboundary_faces)
    abs_dual_coboundary_faces[2] = abs_dual_coboundary_faces[2]**2
    # with this product, we extract with the right orientation the boundary pieces
    diagonal_times_c = spmv.spmm(diagonal_matrix_COO, c.coeffs,
                                 transpose=False,
                                 shape=c.complex.S[n-1].shape[0])
    # here we sum their contribution taking into account the orientation
    d_closure_coeffs = spmv.spmm(abs_dual_coboundary_faces, diagonal_times_c,
                                 transpose=False,
                                 shape=c.complex.num_nodes)
    d_closure = CochainD(dim=n, complex=c.complex, coeffs=0.5*d_closure_coeffs)
    return d_closure


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
        star_c.coeffs = (c.complex.hodge_star[c.dim]*c.coeffs.T).T
    else:
        # NOTE: this step only works with well-centered meshes!
        assert c.complex.is_well_centered
        star_c.coeffs = (
            c.complex.hodge_star_inverse[c.complex.dim - c.dim]*c.coeffs.T).T

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

    if c1.coeffs.ndim == 1:
        assert c2.coeffs.ndim == 1
        inner_product = dt.backend.dot(c1.coeffs, star_c_2.coeffs)
    elif c1.coeffs.ndim == 2:
        assert c2.coeffs.ndim == 2
        # c1_coeffs_T = c1.coeffs.T
        inner_product = dt.backend.sum(c1.coeffs * star_c_2.coeffs)
    elif c1.coeffs.ndim == 3:
        assert c2.coeffs.ndim == 3
        c1_coeffs_T = dt.backend.transpose(c1.coeffs, axes=(0, 2, 1))
        inner_product_per_cell = dt.backend.trace(
            c1_coeffs_T @ star_c_2.coeffs, axis1=1, axis2=2)
        inner_product = dt.backend.sum(inner_product_per_cell)
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


def deformation_gradient(c: Cochain) -> Cochain:
    F = c.complex.get_deformation_gradient(c.coeffs)
    return Cochain(0, not c.is_primal, c.complex, F)


def transpose(c: Cochain) -> Cochain:
    return Cochain(c.dim, c.is_primal, c.complex, jnp.transpose(c.coeffs,
                                                                axes=(0, 2, 1)))


def trace(c: Cochain) -> Cochain:
    return Cochain(c.dim, c.is_primal, c.complex, jnp.trace(c.coeffs, axis1=1, axis2=2))


def vector_tensor_mul(c_v: Cochain, c_T: Cochain) -> Cochain:
    return Cochain(c_T.dim, c_T.is_primal, c_T.complex,
                   c_v.coeffs[:, None, None]*c_T.coeffs)


def sym(c: Cochain) -> Cochain:
    return scalar_mul(add(c, transpose(c)), 0.5)
