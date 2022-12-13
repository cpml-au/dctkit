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
    # c must be a primal 0-cochain
    assert (c.dim == 0 and c.is_primal)

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

    # p must be a dual 2-cochain
    assert (p.dim == 2 and not p.is_primal)

    return p


def Poisson_vec_operator(x, S, k, boundary_values):
    """Discrete laplacian starting from a vector instead of a cochain.

    Args:
        x (np.array): the vector of coefficients of the cochain in which we apply
                      Poisson.
        S (SimplicialComplex): a simplicial complex in which we define the cochain to
                              apply Poisson.
        k (float): The diffusitivity coeffficient.
        boundary_values (tuple): tuple of two np.arrays in which the first encodes the
                                 positions of boundary values, while the last encodes
                                 the boundary values themselves.
    Returns:
        w (np.array): vector of coefficients of the cochain obtained through the
                      discrete laplacian operator.
    """
    c = C.Cochain(0, True, S, x)
    p = Poisson(c, k, boundary_values)
    w = p.coeffs
    return w


def obj_poisson(x, f, S, k, boundary_values, gamma):
    """Objective function of the Poisson optimization problem.

    Args:
        x (np.array): vector in which we want to evaluate the objective function.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the cochain to
                              apply Poisson.
        k (float): The diffusitivity coeffficient.
        boundary_values (tuple): tuple of two np.arrays in which the first encodes the
                                 positions of boundary values, while the last encodes
                                 the boundary values themselves.
        gamma (float): penalty term.
    Returns:
        energy (float): the value of the objective function at x.
    """
    pos, value = boundary_values
    Ax = Poisson_vec_operator(x, S, k, boundary_values)
    r = Ax - f
    # \sum_i (x_i - value_i)^2
    penalty = np.sum((x[pos] - value)**2)
    energy = 0.5*np.linalg.norm(r)**2 + 0.5*gamma*penalty
    return energy


def grad_poisson(x, f, S, k, boundary_values, gamma):
    """Gradient of the objective function of the Poisson optimization problem.

    Args:
        x (np.array): vector in which we want to evaluate the objective function.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the cochain to
                              apply Poisson.
        k (float): The diffusitivity coeffficient.
        boundary_values (tuple): tuple of two np.arrays in which the first encodes the
                                 positions of boundary values, while the last encodes
                                 the boundary values themselves.
        gamma (float): penalty term.
    Returns:
        grad_energy (np.array): the value of the gradient of the objective function at
                                x.
    """
    pos, value = boundary_values
    Ax = Poisson_vec_operator(x, S, k, boundary_values)
    # gradient of the residual = A(Ax - f) since A is symmetric
    grad_r = Poisson_vec_operator(Ax - f, S, k, boundary_values)
    grad_penalty = np.zeros(len(Ax))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_r + gamma*grad_penalty
    return grad_energy
