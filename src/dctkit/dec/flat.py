import jax.numpy as jnp
from dctkit.dec import cochain as C
from jax import Array


def flat(c: C.CochainP0 | C.CochainD0, weights: Array,
         edges: C.CochainP1V | C.CochainD1V) -> C.CochainP1 | C.CochainD1:

    weighted_v = c.coeffs @ weights
    if c.coeffs.ndim == 2:
        # vector field case
        # perform dot product row-wise with the edge vectors
        # of the dual edges (see definition of DPD in Hirani, pag. 54).
        weighted_v_T = weighted_v.T
        coch_coeffs = jnp.einsum("ij, ij -> i", weighted_v_T, edges.coeffs)
    elif c.coeffs.ndim == 3:
        # tensor field case
        # apply each matrix (rows of the multiarray weighted_v_T fixing the first axis)
        # to the edge vector of the corresponding dual edge
        weighted_v_T = jnp.transpose(weighted_v, axes=(2, 0, 1))
        coch_coeffs = jnp.einsum("ijk, ik -> ij", weighted_v_T, edges.coeffs)

    if edges.is_primal:
        return C.CochainP1(c.complex, coch_coeffs)
    else:
        return C.CochainD1(c.complex, coch_coeffs)


def flat_DPD(c: C.CochainD0V | C.CochainD0T) -> C.CochainD1:
    """Implements the flat DPD operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the dual 1-cochain resulting from the application of the flat operator.
    """
    dual_edges = c.complex.dual_edges_vectors[:, :c.coeffs.shape[0]]
    flat_matrix = c.complex.flat_DPD_weights

    return flat(c, flat_matrix, C.CochainD1(c.complex, dual_edges))


def flat_DPP(c: C.CochainD0V | C.CochainD0T) -> C.CochainP1:
    """Implements the flat DPP operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the primal 1-cochain resulting from the application of the flat operator.
    """
    primal_edges = c.complex.primal_edges_vectors[:, :c.coeffs.shape[0]]
    flat_matrix = c.complex.flat_DPP_weights

    return flat(c, flat_matrix, C.CochainP1(c.complex, primal_edges))
