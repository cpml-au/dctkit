import itertools
import jax.numpy as jnp
from jax import Array, vmap
from dctkit.dec import cochain as C
from scipy.special import factorial
from functools import partial
from typing import List


def compute_permutation_vectors(n: int) -> Array:
    """Computes all permutation vectors of length n.

    Args:
        n: The number of elements to permute.

    Returns:
        a JAX array of shape (n!, n) containing all permutations
        of the integers from 0 to n-1.
    """
    perms = list(itertools.permutations(range(n)))
    perm_array = jnp.array(perms)
    return perm_array


@vmap
def permutation_sign(p: Array) -> Array:
    """Computes the sign of a permutation.

    The sign is +1 for even permutations and -1 for odd permutations.
    It is computed as the determinant of the corresponding permutation matrix.

    Args:
        p: A 1D array representing a permutation of integers.

    Returns:
        the sign of the permutations.
    """
    n = len(p)
    # Permutation matrix
    perm_matrix = jnp.eye(n)[p]
    return round(jnp.linalg.det(perm_matrix))


@partial(vmap, in_axes=(0, None))
def find_simplex_idx(s: Array, S: Array) -> Array:
    """Finds the index of a given simplex in a set of simplices.

    Args:
        s: A 1D array representing a simplex (e.g., a set of vertex indices).
        S: A 2D array where each row is a simplex.

    Returns:
        the index of the simplex `s` in `S`. If `s` is not found,
        returns -1.
    """
    # Broadcast and compare all rows to the given row
    matches = jnp.all(S == s, axis=1)
    # Find the index where all elements match
    simplex_idx = jnp.where(matches, size=1, fill_value=-1)[0][0]
    return simplex_idx


@partial(vmap, in_axes=(0, None, None, None, None, None, None))
def compute_wedge_coeffs(simplex: Array,
                         S_list: List[Array],
                         c_1: C.Cochain,
                         c_2: C.Cochain,
                         perm_vec: Array,
                         sgn_perm_vec: Array,
                         weight: Array) -> Array:
    """Computes the coefficients of the wedge product for a simplex.

    This function computes the weighted wedge product of two cochains over
    a simplex, taking into account permutations and orientation signs. It
    is vectorized over the first argument (`simplex`) using `jax.vmap`.

    Args:
        simplex: a 1D array representing the indices of a simplex.
        S_list: a list of arrays, where each array contains all simplices of
            a given dimension.
        c_1: the first cochain object.
        c_2: the second cochain object.
        perm_vec: an array representing a permutation of the simplex indices.
        sgn_perm_vec: the sign (+1 or -1) associated with the permutation.
        weight: a scalar weight to apply to the wedge product.

    Returns:
        the weighted sum of the wedge product contributions for the
        permuted simplex.
    """
    # perm the simplex idx vector
    perm_simplex = simplex[perm_vec]
    # split the perm simplices in vector of indices compatible
    # with c_1 and c_2
    perm_simplex_c_1 = perm_simplex[:, :c_1.dim+1]
    perm_simplex_c_2 = perm_simplex[:, c_1.dim:]

    # since the perm simplices may not have the same orientations
    # as the original one, we need to account for that
    perm_ord_c_1 = jnp.argsort(perm_simplex_c_1, axis=1)
    perm_ord_c_2 = jnp.argsort(perm_simplex_c_2, axis=1)
    sign_orientations_c_1 = permutation_sign(perm_ord_c_1)
    sign_orientations_c_2 = permutation_sign(perm_ord_c_2)

    # compute the indexes for every (ordered) perm_simplex
    ord_simplex_c_1 = jnp.take_along_axis(perm_simplex_c_1, perm_ord_c_1, axis=1)
    ord_simplex_c_2 = jnp.take_along_axis(perm_simplex_c_2, perm_ord_c_2, axis=1)
    perm_idx_c_1 = find_simplex_idx(ord_simplex_c_1, S_list[c_1.dim])
    perm_idx_c_2 = find_simplex_idx(ord_simplex_c_2, S_list[c_2.dim])

    # compute the value of the cup product
    cup_prod_no_sign = c_1.coeffs[perm_idx_c_1]*c_2.coeffs[perm_idx_c_2]
    cup_prod = cup_prod_no_sign.ravel()*sign_orientations_c_1*sign_orientations_c_2

    # compute wedge entry
    wedge_vec = sgn_perm_vec*cup_prod
    return weight*jnp.sum(wedge_vec)


def wedge(c_1: C.Cochain, c_2: C.Cochain) -> C.Cochain:
    """Computes the wedge product of two cochains.

    Args:
        c_1: the first cochain.
        c_2: the second cochain.

    Returns:
        a new cochain representing the wedge product of `c_1` and `c_2`.

    Raises:
        Exception: If attempting a primal-dual wedge product, which is
        undefined.
        AssertionError: If computing a dual wedge product with dimension
        greater than 1, which is not defined.
    """
    wedge_coch_dim = c_1.dim + c_2.dim
    weight = 1/factorial(wedge_coch_dim+1, True)
    S = c_1.complex
    # extract the matrix of indices of the wedge_coch_dim+1-simplices (primal/dual)
    if c_1.is_primal and c_2.is_primal:
        # primal wedge
        S_list = S.S
    elif (not c_1.is_primal) and not (c_2.is_primal):
        # dual wedge is only defined for wedge_coch_dim <=1
        assert wedge_coch_dim <= 1
        S_list = S.S_dual
    else:
        raise Exception("The primal-dual wedge product is not defined.")
    simplices = S_list[wedge_coch_dim]
    # generate the permutation vectors and compute its signs
    perm_vec = compute_permutation_vectors(wedge_coch_dim+1)
    sgn_perm_vec = permutation_sign(perm_vec)
    # compute wedge coeffs
    wedge_coch_coeffs = compute_wedge_coeffs(
        simplices, S_list, c_1, c_2, perm_vec, sgn_perm_vec, weight)
    return C.Cochain(wedge_coch_dim, c_1.is_primal, S, wedge_coch_coeffs)
