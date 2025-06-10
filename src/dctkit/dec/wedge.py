import itertools
import jax.numpy as jnp
from jax import Array, vmap, debug
from dctkit.dec import cochain as C
import dctkit as dt
from jax.scipy.special import factorial
from functools import partial
from dctkit.mesh import util


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


def primal_wedge(c_1: C.CochainP, c_2: C.CochainP) -> C.CochainP:
    wedge_coch_dim = c_1.dim + c_2.dim
    weight = 1/factorial(wedge_coch_dim+1)
    S = c_1.complex
    # extract the matrix of indices of the wedge_coch_dim+1-simplices
    simplices = S.S[wedge_coch_dim]
    # extract the number of wedge_coch_dim-simplices
    num_simplices = simplices.shape[0]
    # generate the permutation vectors and compute its signs
    perm_vec = compute_permutation_vectors(wedge_coch_dim+1)
    sgn_perm_vec = permutation_sign(perm_vec)

    # fill the coeffs of wedge coch
    wedge_coch_coeffs = jnp.zeros(num_simplices, dtype=dt.float_dtype)
    # FIXME: optimize this with vmap
    for i, simplex in enumerate(simplices):
        # perm the simplex idx vector
        perm_simplex = simplex[perm_vec]
        # split the perm simplices in vector of indices compatible
        # with c_1 and c_2
        perm_simplex_c_1 = perm_simplex[:, :c_1.dim+1]
        perm_simplex_c_2 = perm_simplex[:, c_1.dim:]

        print(perm_simplex_c_1, jnp.argsort(perm_simplex_c_1, axis=1))
        assert False

        # compute the indexes for every perm_simplex
        perm_idx_c_1 = find_simplex_idx(perm_simplex_c_1, S.S[c_1.dim])
        perm_idx_c_2 = find_simplex_idx(perm_simplex_c_2, S.S[c_2.dim])

        # compute wedge entry
        wedge_vec = sgn_perm_vec*c_1.coeffs[perm_idx_c_1]*c_2.coeffs[perm_idx_c_2]
        wedge_coch_coeffs = wedge_coch_coeffs.at[i].set(weight*jnp.sum(wedge_vec))

    return C.CochainP(wedge_coch_dim, S, wedge_coch_coeffs)


if __name__ == "__main__":
    mesh_1, _ = util.generate_line_mesh(5, 1.)
    mesh_2, _ = util.generate_square_mesh(0.8)
    mesh_3, _ = util.generate_tet_mesh(2.0)
    S_1 = util.build_complex_from_mesh(mesh_1)
    S_2 = util.build_complex_from_mesh(mesh_2)
    S_3 = util.build_complex_from_mesh(mesh_3)
    S_1.get_hodge_star()
    S_2.get_hodge_star()
    S_3.get_hodge_star()

    print(S_2.S[1])
    assert False

    # vP0_1 = jnp.array([1, 2, 3, 4, 5], dtype=dt.float_dtype)
    # vP0_2 = jnp.array([6, 7, 8, 9, 10], dtype=dt.float_dtype)
    # vP1_1 = jnp.array([1, 2, 3, 4], dtype=dt.float_dtype)
    # vP1_2 = jnp.array([5, 6, 7, 8], dtype=dt.float_dtype)

    # cP0_1 = C.CochainP0(complex=S_1, coeffs=vP0_1)
    # cP0_2 = C.CochainP0(complex=S_1, coeffs=vP0_2)
    # cP1_1 = C.CochainP1(complex=S_1, coeffs=vP1_1)
    # cP1_2 = C.CochainP1(complex=S_1, coeffs=vP1_2)

    vP1_1 = jnp.arange(1, 9, dtype=dt.float_dtype)
    vP1_2 = jnp.arange(8, 17, dtype=dt.float_dtype)

    cP1_1 = C.CochainP1(complex=S_2, coeffs=vP1_1)
    cP1_2 = C.CochainP1(complex=S_2, coeffs=vP1_2)

    print(primal_wedge(cP1_1, cP1_2).coeffs)
