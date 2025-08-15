import itertools
import jax.numpy as jnp
from jax import Array, vmap
from dctkit.dec import cochain as C
from scipy.special import factorial
from functools import partial


def compute_permutation_vectors(n: int) -> Array:
    perms = list(itertools.permutations(range(n)))
    perm_array = jnp.array(perms)
    return perm_array


@vmap
def permutation_sign(p: Array) -> float:
    n = len(p)
    # Permutation matrix
    perm_matrix = jnp.eye(n)[p]
    return round(jnp.linalg.det(perm_matrix))


@partial(vmap, in_axes=(0, None))
def find_simplex_idx(s: Array, S: Array):
    # Broadcast and compare all rows to the given row
    matches = jnp.all(S == s, axis=1)
    # Find the index where all elements match
    simplex_idx = jnp.where(matches, size=1, fill_value=-1)[0][0]
    return simplex_idx


@partial(vmap, in_axes=(0, None, None, None, None, None, None))
def compute_wedge_coeffs(simplex, S_list, c_1, c_2, perm_vec, sgn_perm_vec, weight):
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
