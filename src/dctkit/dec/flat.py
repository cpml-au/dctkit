import jax.numpy as jnp
from dctkit.dec import cochain as C
from jax import Array, vmap
from functools import partial
from typing import Callable, Dict


def flat(c: C.CochainP0 | C.CochainD0, weights: Array, edges: C.CochainP1V |
         C.CochainD1V, I: Callable = None,
         I_args: Dict = {}) -> C.CochainP1 | C.CochainD1:
    """Applies the flat to a vector/tensor-valued cochain representing a discrete
    vector/tensor field to obtain a scalar-valued cochain over primal/dual edges.

    Args:
        c: input vector/tensor-valued 0-cochain representing a primal/dual discrete
            vector/tensor field.
        weights: array of weights that represent the contribution of each component of
            the input cochain to the primal/dual edges where the output cochain is
            defined (i.e. where integration is performed). The number of columns must
            be equal to the number of primal/dual target edges. Weights depend on the
            interpolation scheme chosen for the input discrete vector/tensor field.
        edges: vector-valued cochain collecting the primal/dual edges over which the
            discrete vector/tensor field should be integrated.
    Returns:
        a primal/dual scalar/vector-valued cochain defined over primal/dual edges.
    """
    if I is None:
        # contract over the simplices of the input cochain (last axis of weights, first axis
        # of input cochain coeffs)
        def I(x): return jnp.tensordot(weights.T, x.coeffs, axes=1)
        I_args = {}
    weighted_v = I(c, **I_args)
    # contract input vector/tensors with edge vectors (last indices of both
    # coefficient matrices), for each target primal/dual edge
    contract = partial(jnp.tensordot, axes=([-1,], [-1,]))
    # map over target primal/dual edges
    batch_contract = vmap(contract)
    coch_coeffs = batch_contract(weighted_v, edges.coeffs)

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
    dual_edges = c.complex.dual_edges_vectors[:, :c.coeffs.shape[1]]
    flat_matrix = c.complex.flat_DPD_weights

    return flat(c, flat_matrix, C.CochainD1(c.complex, dual_edges))


def flat_DPP(c: C.CochainD0V | C.CochainD0T) -> C.CochainP1:
    """Implements the flat DPP operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the primal 1-cochain resulting from the application of the flat operator.
    """
    primal_edges = c.complex.primal_edges_vectors[:, :c.coeffs.shape[1]]
    flat_matrix = c.complex.flat_DPP_weights

    return flat(c, flat_matrix, C.CochainP1(c.complex, primal_edges))
