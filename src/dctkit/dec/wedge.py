import jax.numpy as jnp
from jax import Array, vmap
from dctkit.dec import cochain as C
from dctkit.math.permutation import *
from scipy.special import factorial
from functools import partial
from typing import List


# @partial(vmap, in_axes=(0, None, None, None, None, None, None))
# def compute_wedge_coeffs(simplex: Array,
#                          S_list: List[Array],
#                          c_1: C.Cochain,
#                          c_2: C.Cochain,
#                          perm_vec: Array,
#                          sgn_perm_vec: Array,
#                          weight: Array) -> Array:
#     """Computes the coefficients of the wedge product for a simplex.

#     This function computes the weighted wedge product of two cochains over
#     a simplex, taking into account permutations and orientation signs. It
#     is vectorized over the first argument (`simplex`) using `jax.vmap`.

#     Args:
#         simplex: a 1D array representing the indices of a simplex.
#         S_list: a list of arrays, where each array contains all simplices of
#             a given dimension.
#         c_1: the first cochain object.
#         c_2: the second cochain object.
#         perm_vec: an array representing a permutation of the simplex indices.
#         sgn_perm_vec: the sign (+1 or -1) associated with the permutation.
#         weight: a scalar weight to apply to the wedge product.

#     Returns:
#         the weighted sum of the wedge product contributions for the
#         permuted simplex.
#     """
#     # # perm the simplex idx vector
#     # perm_simplex = simplex[perm_vec]
#     # # split the perm simplices in vector of indices compatible
#     # # with c_1 and c_2
#     # perm_simplex_c_1 = perm_simplex[:, :c_1.dim+1]
#     # perm_simplex_c_2 = perm_simplex[:, c_1.dim:]

#     # # since the perm simplices may not have the same orientations
#     # # as the original one, we need to account for that
#     # perm_ord_c_1 = jnp.argsort(perm_simplex_c_1, axis=1)
#     # perm_ord_c_2 = jnp.argsort(perm_simplex_c_2, axis=1)
#     # sign_orientations_c_1 = permutation_sign(perm_ord_c_1)
#     # sign_orientations_c_2 = permutation_sign(perm_ord_c_2)

#     # # compute the indexes for every (ordered) perm_simplex
#     # ord_simplex_c_1 = jnp.take_along_axis(perm_simplex_c_1, perm_ord_c_1, axis=1)
#     # ord_simplex_c_2 = jnp.take_along_axis(perm_simplex_c_2, perm_ord_c_2, axis=1)
#     # perm_idx_c_1 = find_simplex_idx(ord_simplex_c_1, S_list[c_1.dim])
#     # perm_idx_c_2 = find_simplex_idx(ord_simplex_c_2, S_list[c_2.dim])

#     # compute the value of the cup product
#     cup_prod_no_sign = c_1.coeffs[perm_idx_c_1]*c_2.coeffs[perm_idx_c_2]
#     cup_prod = cup_prod_no_sign.ravel()*sign_orientations_c_1*sign_orientations_c_2
#     wedge_vec = sgn_perm_vec*cup_prod
#     # FIXME: fix this part of the code
#     if c_1.dim + c_2.dim > 1:
#         weight = weight[0]
#         return weight*jnp.sum(wedge_vec)

#     weight_coeffs = weight[perm_idx_c_2[0]]

#     weighted_wedge_vec = weight_coeffs*wedge_vec[0] + (1-weight_coeffs)*wedge_vec[1]

#     return weighted_wedge_vec


def wedge(c_1: C.Cochain, c_2: C.Cochain, weight: Array = None) -> C.Cochain:
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
    S = c_1.complex
    # extract the matrix of indices of the wedge_coch_dim+1-simplices (primal/dual)
    if c_1.is_primal and c_2.is_primal:
        # primal wedge
        S_list = S.S
        type_ = "primal"
    elif (not c_1.is_primal) and not (c_2.is_primal):
        # dual wedge is only defined for wedge_coch_dim <=1
        assert wedge_coch_dim <= 1
        S_list = S.S_dual
        type_ = "dual"
    else:
        raise Exception("The primal-dual wedge product is not defined.")
    num_c_2_dim_simplex = S_list[c_2.dim].shape[0]
    if weight is None:
        # standard definition
        weight = 1/(wedge_coch_dim+1)*jnp.ones(num_c_2_dim_simplex)
    weight *= 1/factorial(wedge_coch_dim, True)
    # extract the permutation vectors and compute its signs
    sgn_perm_vec = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["sgn_perm_vec"]
    # extract cup product and sgn orientations tables
    lookup = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["lookup"]
    sgn_orient = S.cup_lookup[(type_, c_1.dim, c_2.dim)]["sgn_orient"]

    # compute the value of the cup product
    cup_prod_no_sign = c_1.coeffs[lookup[:, 0]]*c_2.coeffs[lookup[:, 1]]
    cup_prod = cup_prod_no_sign.ravel()*sgn_orient[:, 0]*sgn_orient[:, 1]
    wedge_vec = sgn_perm_vec*cup_prod
    # FIXME: fix this part of the code
    if c_1.dim + c_2.dim > 1:
        weight = weight[0]
        wedge_coch_coeffs = weight*jnp.sum(wedge_vec)
    else:
        weight_coeffs = weight[sgn_orient[:, 1][0]]
        wedge_coch_coeffs = weight_coeffs*wedge_vec[0] + (1-weight_coeffs)*wedge_vec[1]

    # compute wedge coeffs
    # wedge_coch_coeffs = compute_wedge_coeffs(
    #    simplices, S_list, c_1, c_2, perm_vec, sgn_perm_vec, weight)
    return C.Cochain(wedge_coch_dim, c_1.is_primal, S, wedge_coch_coeffs)
