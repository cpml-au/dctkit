import numpy as np
from dctkit.dec import cochain as C


# @profile
def poisson(c, k):
    """Implements a routine to compute the LHS of the Poisson equation in DEC framework.

    Args:
        c (Cochain): A primal 0-cochain.
        k (float): The diffusitivity coefficient.
    Returns:
        Cochain: A dual 2-cochain obtained from the application of the
        discrete laplacian.
    """
    # c must be a primal 0-cochain
    assert (c.dim == 0 and c.is_primal)

    # compute the coboundary
    c_1 = C.coboundary(c)

    # compute star
    star_c_1 = C.star(c_1)

    # constitutive relation for the flux
    h = C.Cochain(star_c_1.dim, star_c_1.is_primal, star_c_1.complex,
                  -k*star_c_1.coeffs)

    # coboundary again to obtain a dual 2-cochain
    p = C.coboundary(h)

    # p must be a dual 2-cochain
    assert (p.dim == 2 and not p.is_primal)

    return p


# @profile
def poisson_vec_operator(x, S, k):
    """Discrete laplacian starting from a vector instead of a cochain.

    Args:
        x (np.array): the vector of coefficients of the cochain in which we
            apply Poisson.
        S (SimplicialComplex): a simplicial complex in which we define Poisson.
        k (float): the diffusitivity coefficient.
    Returns:
        np.array: vector of coefficients of the cochain obtained through the
        discrete laplacian operator.
    """
    c = C.Cochain(0, True, S, x)
    p = poisson(c, k)
    w = p.coeffs
    return w


# @profile
def obj_poisson(x, f, S, k, boundary_values, gamma, mask):
    """Objective function of the Poisson optimization problem.

    Args:
        x (np.array): vector in which we want to evaluate the objective function.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the
            cochain to apply Poisson.
        k (float): the diffusitivity coefficient.
        boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma (float): penalty factor.
    Returns:
        float: the value of the objective function at x.
    """
    pos, value = boundary_values
    Ax = poisson_vec_operator(x, S, k)
    r = Ax - f
    # zero residual on dual cells at the boundary where nodal values are imposed
    # r_proj = r.at[pos].set(0.)

    # \sum_i (x_i - value_i)^2
    penalty = np.sum((x[pos] - value)**2)
    energy = 0.5*np.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
    return energy


# @profile
def grad_poisson(x, f, S, k, boundary_values, gamma, mask):
    """Gradient of the objective function of the Poisson optimization problem.

    Args:
        x (np.array): vector in which we want to evaluate the gradient.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the
            cochain to apply Poisson.
        k (float): the diffusitivity coefficient.
        boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma (float): penalty factor.
    Returns:
        np.array: the value of the gradient of the objective function at x.
    """
    pos, value = boundary_values
    Ax = poisson_vec_operator(x, S, k)
    r = Ax - f
    # zero residual on dual cells at the boundary where nodal values are imposed
    # r_proj = r.at[pos].set(0.)

    # gradient of the projected residual = A^T r_proj = A r_proj, since A is symmetric
    grad_r = poisson_vec_operator(r*mask, S, k)
    grad_penalty = np.zeros(len(Ax))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_r + gamma*grad_penalty
    return grad_energy


def energy_poisson(x, f, S, k, boundary_values, gamma):
    """Implementation of the Dirichlet energy.

    Args:
        x (np.array): vector in which we want to evaluate the objective function.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the
            cochain to apply Poisson.
        k (float): the diffusitivity coefficient.
        boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma (float): penalty factor.
    Returns:
        float: the value of the objective function at x.
    """
    pos, value = boundary_values
    f = C.Cochain(0, True, S, f)
    u = C.Cochain(0, True, S, x)
    du = C.coboundary(u)
    norm_grad = k/2*C.inner_product(du, du)
    bound_term = C.inner_product(u, f)
    penalty = 0.5*gamma*np.sum((x[pos] - value)**2)
    energy = norm_grad + bound_term + penalty
    return energy


def grad_energy_poisson(x, f, S, k, boundary_values, gamma):
    """Gradient of the Dirichlet energy.

    Args:
        x (np.array): vector in which we want to evaluate the gradient.
        f (np.array): vector of external sources (constant term of the system).
        S (SimplicialComplex): a simplicial complex in which we define the
            cochain to apply Poisson.
        k (float): the diffusitivity coefficient.
        boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma (float): penalty factor.
    Returns:
        np.array: the value of the gradient of the objective function at x.
    """
    pos, value = boundary_values
    f = C.Cochain(0, True, S, f)
    # grad_1(x) = 1/2 * (A + A^T) x = Ax since A=A^T
    grad_1 = -poisson_vec_operator(x, S, k)
    star_f = C.star(f)
    # grad_2(x) = Hx, where H is the hodge star matrix
    grad_2 = star_f.coeffs
    grad_penalty = np.zeros(len(grad_1))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_1 + grad_2 + gamma*grad_penalty
    return grad_energy
