import itertools
import jax.numpy as jnp
from jax import Array
from dctkit.dec import cochain as C
import dctkit as dt
from jax.scipy.special import factorial


def compute_permutation_vectors(n: int) -> Array:
    perms = list(itertools.permutations(range(n)))
    perm_array = jnp.array(perms)
    return perm_array


def primal_wedge(c_1: C.CochainP, c_2: C.CochainP) -> C.CochainP:
    wedge_coch_dim = c_1.dim + c_2.dim
    weight = 1/factorial(wedge_coch_dim+1)
    # extract the matrix of indices of the wedge_coch_dim+1-simplices
    S_klp1 = c_1.S.S[wedge_coch_dim+1]
    # extract the number of wedge_coch_dim+1-simplices
    num_simplices = S_klp1.shape[0]
    # generate the permutation vectors
    perm_vectors = compute_permutation_vectors(num_simplices)

    # fill the coeffs of wedge coch
    wedge_coch_coeffs = jnp.zeros(num_simplices, dtype=dt.float_dtype)
    # FIXME: optimize this with vmap
    for i, simplex_idx in enumerate(S_klp1):
        pass
    return C.CochainP(wedge_coch_dim, c_1.S, wedge_coch_coeffs)


if __name__ == "__main__":
    print(compute_permutation_vectors(4).shape)
