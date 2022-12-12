import numpy as np
from dctkit.dec import cochain as C


def Poisson(c, k, boundary_values):
    """Implements a routine to compute the LHS of the Poisson equation in DEC framework.

    Args:
        c (Cochain): A primal 0-cochain.
        k (float): The diffusitivity coeffficient.
        boundary_values (tuple): tuple of two np.arrays in which the first encodes the
                                 positions of boundary values, while the last encodes
                                 the boundary values themselves.
    Returns:
        p (Cochain): A dual 2-cochain obtained from the application of the discrete
                     laplacian.
    """
    # extract boundary values and their positions
    pos, value = boundary_values

    # create a new cochain that takes in account boundary values
    c_new = C.Cochain(c.dim, c.is_primal, c.complex,
                      np.empty(len(pos) + len(c.coeffs)))
    # create a mask to track indexes other than pos
    mask = np.ones(len(pos) + len(c.coeffs), bool)
    mask[pos] = False
    c_new.coeffs[pos] = value
    c_new.coeffs[mask] = c.coeffs
    c = c_new

    # compute the coboundary
    c_1 = C.coboundary(c)
    # compute star
    star_c_1 = C.star(c_1)
    # contitutive relation
    h = C.Cochain(star_c_1.dim, star_c_1.is_primal, star_c_1.complex,
                  -k*star_c_1.coeffs)
    # coboundary again to obtain a dual 2-cochain
    p = C.coboundary(h)
    return p
