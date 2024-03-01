import jax.numpy as jnp
import dctkit as dt
from dctkit.dec import cochain as C


def flat_DPD(c: C.CochainD0V | C.CochainD0T) -> C.CochainD1:
    """Implements the flat DPD operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the dual 1-cochain resulting from the application of the flat operator.
    """
    dedges = c.complex.dual_edges_vectors[:, :c.coeffs.shape[0]]
    flat_matrix = c.complex.flat_DPD_weights
    # multiply weights of each dual edge by the vectors associated to the dual nodes
    # belonging to the edge
    weighted_v = c.coeffs @ flat_matrix
    if c.coeffs.ndim == 2:
        # vector field case
        # perform dot product row-wise with the edge vectors
        # of the dual edges (see definition of DPD in Hirani, pag. 54).
        weighted_v_T = weighted_v.T
        coch_coeffs = jnp.einsum("ij, ij -> i", weighted_v_T, dedges)
    elif c.coeffs.ndim == 3:
        # tensor field case
        # apply each matrix (rows of the multiarray weighted_v_T fixing the first axis)
        # to the edge vector of the corresponding dual edge
        weighted_v_T = jnp.transpose(weighted_v, axes=(2, 0, 1))
        coch_coeffs = jnp.einsum("ijk, ik -> ij", weighted_v_T, dedges)
    return C.CochainD1(c.complex, coch_coeffs)


def flat_DPP(c: C.CochainD0V | C.CochainD0T) -> C.CochainP1:
    """Implements the flat DPP operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the primal 1-cochain resulting from the application of the flat operator.
    """
    primal_edges = c.complex.primal_edges_vectors[:, :c.coeffs.shape[0]]
    flat_matrix = c.complex.flat_DPP_weights
    # multiply weights of each primal edge by the vectors associated to the dual nodes
    # belonging to the corresponding dual edge
    weighted_v = c.coeffs @ flat_matrix
    if c.coeffs.ndim == 2:
        # vector field case
        # perform dot product row-wise with the edge vectors
        # of the dual edges (see definition of DPD in Hirani, pag. 54).
        weighted_v_T = weighted_v.T
        coch_coeffs = jnp.einsum("ij, ij -> i", weighted_v_T,
                                 primal_edges)
    elif c.coeffs.ndim == 3:
        # tensor field case
        # apply each matrix (rows of the multiarray weighted_v_T fixing the first axis)
        # to the edge vector of the corresponding dual edge
        weighted_v_T = jnp.transpose(weighted_v, axes=(2, 0, 1))
        coch_coeffs = jnp.einsum("ijk, ik -> ij", weighted_v_T,
                                 primal_edges)
    return C.CochainP1(c.complex, coch_coeffs)


def flat_PDP(c: C.CochainP0) -> C.CochainP1:
    return C.CochainP1(c.complex, c.complex.flat_PDP_weights @ c.coeffs)


def flat_PDD(c: C.CochainD0, scheme: str) -> C.CochainD1:
    # NOTE: we use periodic boundary conditions
    # NOTE: only implemented for dim = 1, where dim is the dimension
    # of the complex
    dual_volumes = c.complex.dual_volumes[0]
    # FIXME: rewrite this!
    if c.coeffs.ndim == 1:
        flat_c_coeffs = jnp.zeros(c.complex.num_nodes, dtype=dt.float_dtype)
    elif c.coeffs.ndim == 2:
        flat_c_coeffs = jnp.zeros(
            (c.complex.num_nodes, c.coeffs.shape[1]), dtype=dt.float_dtype)
    if scheme == "upwind":
        # periodic bc
        flat_c_coeffs = flat_c_coeffs.at[0].set(dual_volumes[0]*c.coeffs[-1])
        # upwind implementation
        flat_c_coeffs = flat_c_coeffs.at[1:].set(dual_volumes[1:]*c.coeffs)
    elif scheme == "parabolic":
        # periodic bc
        flat_c_coeffs = flat_c_coeffs.at[0].set(dual_volumes[0]*c.coeffs[-1])
        flat_c_coeffs = flat_c_coeffs.at[-1].set(dual_volumes[-1]*c.coeffs[0])
        flat_c_coeffs = flat_c_coeffs.at[1:-1].set(0.5 * (dual_volumes[1:-1] *
                                                   (c.coeffs[:-1] + c.coeffs[1:]).T).T)
    return C.CochainD1(c.complex, flat_c_coeffs)
