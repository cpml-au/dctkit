import dctkit.dec.cochain as C
from dctkit.math import spmm
import jax.numpy as jnp


def coboundary_closure(c: C.CochainP) -> C.CochainD:
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
    ones = C.CochainD(dim=n-2, complex=c.complex, coeffs=jnp.ones(num_tets))
    diagonal_elems = C.coboundary(ones).coeffs.flatten()
    diagonal_matrix_rows = jnp.arange(num_dual_faces)
    diagonal_matrix_cols = diagonal_matrix_rows
    diagonal_matrix_COO = [diagonal_matrix_rows, diagonal_matrix_cols, diagonal_elems]

    # build the absolute value of the (n-1)-coboundary
    abs_dual_coboundary_faces = c.complex.boundary[n-1].copy()
    # same of doing abs(dual_coboundary_faces)
    abs_dual_coboundary_faces[2] = abs_dual_coboundary_faces[2]**2
    # with this product, we extract with the right orientation the boundary pieces
    diagonal_times_c = spmm.spmm(diagonal_matrix_COO, c.coeffs,
                                 transpose=False,
                                 shape=c.complex.S[n-1].shape[0])
    # here we sum their contribution taking into account the orientation
    d_closure_coeffs = spmm.spmm(abs_dual_coboundary_faces, diagonal_times_c,
                                 transpose=False,
                                 shape=c.complex.num_nodes)
    d_closure = C.CochainD(dim=n, complex=c.complex, coeffs=0.5*d_closure_coeffs)
    return d_closure
