import dctkit as dt
import jax.numpy as jnp
import numpy.typing as npt
from .cochain import CochainD1
from dctkit.mesh import simplex as spx
from jax import Array


class DiscreteVectorField():
    """Discrete vector fields class.

    Args:
        S (SimplicialComplex): the simplicial complex where the discrete vector field
        is defined.
        is_primal (bool): True if the discrete vector field is primal, False otherwise.
        coeffs (Array): array of the coefficients of the discrete vector fields.
    """

    def __init__(self, S: spx.SimplicialComplex, is_primal: bool,
                 coeffs: npt.NDArray | Array):
        self.S = S
        self.is_primal = is_primal
        self.coeffs = coeffs


class DiscreteVectorFieldD(DiscreteVectorField):
    """Inherited class for dual discrete vector fields."""

    def __init__(self, S: spx.SimplicialComplex, coeffs: npt.NDArray | Array):
        super().__init__(S, False, coeffs)


def flat(v: DiscreteVectorFieldD) -> CochainD1:
    """Implements the flat operator for dual discrete vector fields.

    Args:
        v (DiscreteVectorFieldD): a dual discrete vector field.
    Returns:
        (CochainD1): the dual one cochain equal to flat(v).
    """
    dedges = v.S.dual_edges_vectors
    num_dedges = dedges.shape[0]
    flat_matrix = v.S.flat_weights
    coch_coeffs = jnp.zeros(num_dedges, dtype=dt.float_dtype)
    for i in range(num_dedges):
        # extract indices with non-zero entries in the flat matrix.
        good_indices = flat_matrix[:, i] >= 0
        # normalize right entries
        norm_v_good = (flat_matrix[good_indices, i] * v.coeffs[good_indices, :].T).T
        # i-th entry of coch_coeffs is the sum of the entries of the vector obtained
        # by multiplying the matrix norm_v with the coords of the i-th dual edge
        coch_coeffs = coch_coeffs.at[i].set(jnp.sum(norm_v_good @ dedges[i, :]))
    return CochainD1(v.S, coch_coeffs)
