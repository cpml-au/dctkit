import numpy as np
from dctkit.dec import cochain as C
from dctkit.mesh import simplex
from typing import Tuple
import numpy.typing as npt
from jax import Array


def poisson_residual(u: C.CochainP0, f: C.CochainD2, k: float) -> C.Cochain:
    """Compute the residual of the discrete Poisson equation in 2D using DEC framework.

        -dh + f = 0,    h = -k*star*du
        => Au + f = 0, A=d star d: matrix associated to the conformal Laplacian

    Args:
        u: a primal 0-cochain
        f: dual 2-cochain of sources
        k: the diffusitivity coefficient
    Returns:
        residual cochain
    """
    # u must be a primal 0-cochain
    assert (u.dim == 0 and u.is_primal)

    # Fick's law for flux and star to get normal flux (across dual 2-cell boundaries)
    du = C.coboundary(u)
    h = C.scalar_mul(C.star(du), -k)

    # net OUTWARD flux (see induced orientation of the dual) across dual 2-cell
    # boundaries
    dh = C.coboundary(h)

    # change sign in order to have INWARD FLUX
    dh = C.scalar_mul(dh, -1.)

    return C.add(dh, f)


def obj_poisson(x: npt.NDArray, f: npt.NDArray, S: simplex.SimplicialComplex, k: float,
                boundary_values: Tuple[npt.NDArray, npt.NDArray], gamma: float,
                mask) -> float:
    """Objective function of the optimization problem associated to Poisson equation
    with Dirichlet boundary conditions.

    Args:
        x: vector in which we want to evaluate the objective function.
        f: vector of external sources (constant term of the system).
        S: a simplicial complex in which we define the
            cochain to apply Poisson.
        k: the diffusitivity coefficient.
        boundary_values: tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma: penalty factor.
    Returns:
        the value of the objective function at x.
    """
    pos, value = boundary_values
    u = C.CochainP0(S, x)
    f_coch = C.CochainD2(S, f)

    r = poisson_residual(u, f_coch, k).coeffs

    penalty = np.sum((x[pos] - value)**2)

    # use mask to zero residual on dual cells at the boundary where nodal values are
    # imposed
    energy = 0.5*np.linalg.norm(r*mask)**2 + 0.5*gamma*penalty
    return energy


def grad_obj_poisson(x: npt.NDArray, f: npt.NDArray, S: simplex.SimplicialComplex,
                     k: float, boundary_values:
                     Tuple[npt.NDArray, npt.NDArray], gamma: float,
                     mask: npt.NDArray) -> Array | npt.NDArray:
    """Gradient of the objective function of the Poisson optimization problem.

    Args:
        x: vector in which we want to evaluate the gradient.
        f: vector of external sources (constant term of the system).
        S: a simplicial complex in which we define the
            cochain to apply Poisson.
        k: the diffusitivity coefficient.
        boundary_values (tuple): tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma: penalty factor.
    Returns:
        the value of the gradient of the objective function at x.
    """
    pos, value = boundary_values
    u = C.CochainP0(S, x)
    f_coch = C.CochainD2(S, f)
    r = poisson_residual(u, f_coch, k).coeffs
    # zero residual on dual cells at the boundary where nodal values are imposed
    r_proj = C.CochainP0(S, r*mask)
    # gradient of the projected residual = A^T r_proj = A r_proj, since A is symmetric
    grad_r = (C.sub(poisson_residual(r_proj, f_coch, k), f_coch)).coeffs
    grad_penalty = np.zeros(len(grad_r))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_r.flatten() + gamma*grad_penalty
    return grad_energy


def energy_poisson(x: npt.NDArray, f: npt.NDArray, S, k: float, boundary_values:
                   Tuple[npt.NDArray, npt.NDArray], gamma: float) -> float:
    """Implementation of the discrete Dirichlet energy.

    Args:
        x: array of the nodal values of the state variable (0-cochain)
        f: array of the coefficients of the primal 0-cochain source term
        S: a simplicial complex representing the mesh where the energy is defined
        k: the diffusitivity coefficient
        boundary_values: tuple of two arrays where the first encodes the positions of
            boundary values, and the second encodes the boundary values
        gamma: penalty factor for the Dirichlet boundary conditions
    Returns:
        value of the energy.
    """
    pos, value = boundary_values
    f_coch = C.CochainP0(S, f)
    u = C.CochainP0(S, x)
    du = C.coboundary(u)
    norm_grad = k/2*C.inner(du, du)
    bound_term = -C.inner(u, f_coch)
    penalty = 0.5*gamma*np.sum((x[pos] - value)**2)
    energy = norm_grad + bound_term + penalty
    return energy


def grad_energy_poisson(x, f, S, k, boundary_values, gamma):
    """Gradient of the Dirichlet energy.

    Args:
        x: vector in which we want to evaluate the gradient.
        f: array of the coefficients of the primal 0-cochain source term
        S: a simplicial complex in which we define the
            cochain to apply Poisson.
        k: the diffusitivity coefficient.
        boundary_values: tuple of two np.arrays in which the first
            encodes the positions of boundary values, while the last encodes the
            boundary values themselves.
        gamma: penalty factor.
    Returns:
        the value of the gradient of the objective function at x.
    """
    pos, value = boundary_values
    u = C.CochainP0(S, x)
    f_coch = C.CochainP0(S, f)
    star_f = C.star(f_coch)

    grad_r = -poisson_residual(u, star_f, k).coeffs

    grad_penalty = np.zeros(len(grad_r))
    grad_penalty[pos] = x[pos] - value
    grad_energy = grad_r.flatten() + gamma*grad_penalty
    return grad_energy
