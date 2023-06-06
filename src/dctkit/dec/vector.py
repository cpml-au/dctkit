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


def flat_DPD(v: DiscreteVectorFieldD) -> CochainD1:
    """Implements the flat operator for dual discrete vector fields.

    Args:
        v: a dual discrete vector field.
    Returns:
        the dual cochain resulting from the application of the flat operator.
    """
    dedges = v.S.dual_edges_vectors
    flat_matrix = v.S.flat_weights
    # multiply weights of each dual edge by the vectors associated to the dual nodes
    # belonging to the edge and then perform dot product row-wise with the edge vectors
    # of the dual edges (see definition of DPD in Hirani, pag. 54).
    coch_coeffs = jnp.sum((v.coeffs @ flat_matrix).T * dedges, axis=1)
    return CochainD1(v.S, coch_coeffs)
