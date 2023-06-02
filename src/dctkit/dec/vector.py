import dctkit as dt
import numpy as np
import jax.numpy as jnp
import numpy.typing as npt
from .cochain import CochainD1
from dctkit.mesh import simplex as spx
from jax import Array


class DiscreteVectorField():
    def __init__(self, S: spx.SimplicialComplex, is_primal: bool,
                 coeffs: npt.NDArray | Array):
        self.S = S
        self.is_primal = is_primal
        self.coeffs = coeffs


class DiscreteVectorFieldD(DiscreteVectorField):
    def __init__(self, S: spx.SimplicialComplex, coeffs: npt.NDArray | Array):
        super().__init__(S, False, coeffs)


def flat(v: DiscreteVectorFieldD) -> CochainD1:
    # FIXME: ADD DOCUMENTATION
    dedges = v.S.dedges
    num_dedges = dedges.shape[0]
    flat_matrix = v.S.flat_coeffs_matrix
    coch_coeffs = np.zeros(num_dedges, dtype=dt.float_dtype)
    for i in range(num_dedges):
        # extract indices with non-zero entries in the flat matrix.
        good_indices = flat_matrix[:, i] >= 0
        # normalize right entries
        norm_v_good = (flat_matrix[good_indices, i] * v.coeffs[good_indices, :].T).T
        # i-th entry of coch_coeffs is the sum of the entries of the vector obtained
        # by multiplying the matrix norm_v with the coords of the i-th dual edge
        print(norm_v_good)
        print(dedges[i, :])
        print("------------------")
        print(norm_v_good @ dedges[i, :])
        coch_coeffs[i] = jnp.sum(norm_v_good @ dedges[i, :])
    print("-------------------------------------")
    return CochainD1(v.S, coch_coeffs)
