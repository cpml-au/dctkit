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
                      np.empty(len(c.coeffs) + len(pos)))
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


def Poisson_vec_operator(x, S, k, boundary_values):
    # S is a predefined SimplicialComplex object
    c = C.Cochain(0, True, S, x)
    p = Poisson(c, k, boundary_values)
    w = p.coeffs
    return w


def obj_poisson(x, f, S, k, boundary_values, gamma):
    pos, value = boundary_values
    Ax = Poisson_vec_operator(x, S, k, boundary_values)
    r = Ax - f
    penalty = np.sum((x[pos] - value)**2)
    energy = 0.5*np.linalg.norm(r)**2 + 0.5*gamma*penalty
    return energy


def grad_poisson(x, f, S, k, boundary_values, gamma):
    pos, value = boundary_values
    Ax = Poisson_vec_operator(x, S, k, boundary_values)
    grad_r = Poisson_vec_operator(Ax - f, S, k, boundary_values)
    grad_penalty = np.zeros(len(Ax))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_r + gamma*grad_penalty
    return grad_energy
