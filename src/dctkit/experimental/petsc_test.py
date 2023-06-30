import petsc4py
from petsc4py import PETSc
import numpy as np
from jax import grad, jit, value_and_grad
from dctkit.mesh import util
import dctkit as dt
from dctkit.dec import cochain as C
import time

dt.config()

lc = 0.008

mesh, _ = util.generate_square_mesh(lc)
S = util.build_complex_from_mesh(mesh)
S.get_hodge_star()
bnodes = mesh.cell_sets_dict["boundary"]["line"]
node_coord = S.node_coords

# NOTE: exact solution of Delta u + f = 0
u_true = np.array(node_coord[:, 0]**2 + node_coord[:, 1] ** 2, dtype=dt.float_dtype)
b_values = u_true[bnodes]

boundary_values = (np.array(bnodes, dtype=dt.int_dtype), b_values)

num_nodes = S.num_nodes
print(num_nodes)
f_vec = -4.*np.ones(num_nodes, dtype=dt.float_dtype)

u_0 = np.zeros(num_nodes, dtype=dt.float_dtype)

gamma = 1000.


def energy_poisson(x, f, boundary_values, gamma):
    pos, value = boundary_values
    f = C.Cochain(0, True, S, f)
    u = C.Cochain(0, True, S, x)
    du = C.coboundary(u)
    norm_grad = 1/2.*C.inner_product(du, du)
    bound_term = -C.inner_product(u, f)
    penalty = 0.5*gamma*dt.backend.sum((x[pos] - value)**2)
    energy = norm_grad + bound_term + penalty
    return energy


args = (f_vec, boundary_values, gamma)

energy_jit = jit(energy_poisson)
objgrad = jit(grad(energy_poisson))
objandgrad = jit(value_and_grad(energy_poisson))
# wrappers for the objective function and its gradient following the signature of
# TAOObjectiveFunction


def objective_function(tao, x, f, boundary_values, gamma):
    return energy_jit(x.getArray(), f, boundary_values, gamma)


def gradient_function(tao, x, g, f, boundary_values, gamma):
    g_jax = objgrad(x.getArray(), f, boundary_values, gamma)
    g.setArray(g_jax)


def objective_and_gradient(tao, x, g, f, boundary_values, gamma):
    fval, grad_jax = objandgrad(x.getArray(), f, boundary_values, gamma)
    g.setArray(grad_jax)
    return fval


# Initialize petsc4py
petsc4py.init()

# Create a PETSc vector to hold the optimization variables
x = PETSc.Vec().createWithArray(u_0)

# Create a PETSc TAO object for the solver
tao = PETSc.TAO().create()
tao.setType(PETSc.TAO.Type.LMVM)  # Specify the solver type
tao.setSolution(x)
# tao.setObjective(objective_function, args=args)  # Set the objective function
g = PETSc.Vec().createSeq(num_nodes)
# tao.setGradient(gradient_function, g, args=args)  # Set the gradient function
tao.setObjectiveGradient(objective_and_gradient, g, args=args)
tao.setMaximumIterations(500)
# tao.setTolerances(gatol=1e-3)
tao.setFromOptions()  # Set options for the solver

tic = time.time()
# Minimize the function using the Nonlinear CG method
tao.solve()
toc = time.time()
tao.view()
print("Elapsed time = ", toc - tic)

# Get the solution and the objective value
u = tao.getSolution()
objective_value = tao.getObjectiveValue()

assert np.allclose(u, u_true, atol=1e-2)
