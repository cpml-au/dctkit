import itertools
import jax.numpy as jnp
from jax import vmap, Array


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
