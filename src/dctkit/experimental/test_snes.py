import petsc4py
from petsc4py.PETSc import SNES, KSP, PC, Mat, Log
from petsc4py.PETSc import Vec, Mat
import numpy as np
from dctkit.mesh import util
import dctkit as dt
from dctkit.dec import cochain as C
import time
from jax import jit, grad, jacrev
import jax.numpy as jnp

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

boundary_values = (np.array(bnodes, dtype=np.int32), b_values)

num_nodes = S.num_nodes
print(num_nodes)
f_vec = -4.*np.ones(num_nodes, dtype=dt.float_dtype)
f = C.CochainP0(S, f_vec)
star_f = C.star(f)

u_0 = np.zeros(num_nodes, dtype=dt.float_dtype)
# u_0 = u_true

gamma = 1000.
k = 1.

kargs = {'f': star_f.coeffs, 'boundary_values': boundary_values, 'gamma': gamma}


@jit
def residual_poisson(x, star_f):
    u = C.CochainP0(S, x)
    f = C.CochainP0(S, star_f)
    du = C.coboundary(u)
    h = C.scalar_mul(C.star(du), -k)

    # net OUTWARD flux (see induced orientation of the dual) across dual 2-cell
    # boundaries
    dh = C.coboundary(h)

    # change sign in order to have INWARD FLUX
    dh = C.scalar_mul(dh, -1.)

    res = C.add(dh, f).coeffs
    res = res.at[boundary_values[0]].set(0.)
    return res


jacobian_poisson = jit(jacrev(residual_poisson))


def residual_poisson_wrap(snes, x: Vec, r: Vec, f, boundary_values, gamma):
    print("residual eval...")
    x_array = x.getArray()
    x_array[boundary_values[0]] = boundary_values[1]
    res = residual_poisson(x_array, f)
    r.setArray(res)
    print("done")


def jacobian_poisson_wrap(snes, x: Vec, J: Mat, P: Mat, f, boundary_values, gamma):
    jac = jacobian_poisson(x.getArray(), f)
    J.setValues(range(0, jac.shape[0]), range(0, jac.shape[1]), jac.flatten())
    print("assembling...")
    J.assemble()
    J.zeroRows(boundary_values[0])
    print("done")

# num_nodes = 2
# x0 = np.zeros(num_nodes)
# # x0[0] = 1.
# # x0[1] = 3.

# def residual(x):
#     return jnp.array([x[0]-1, x[1]-3.])


# def residual_wrap(snes, x: Vec, r: Vec):
#     res = residual(x.getArray())
#     r.setArray(res)


# jacobian = jit(jacrev(residual))


# def jacobian_wrap(snes, x: Vec, J: Mat, P: Mat):
#     print("jac")
#     jac = jacobian(x.getArray())
#     J.setValues(range(0, jac.shape[0]), range(0, jac.shape[1]), jac.flatten())
#     J.assemble()

def monitor(snes, i: int, f: float):
    print("Iteration {}, residual norm = {}".format(i, f))


petsc4py.init()

log = Log()
log.begin()

solver = SNES().create()
solver.setType(SNES.Type.KSPONLY)
solver.setMonitor(monitor)
ksp = KSP().create()
ksp.setType(KSP.Type.CG)
ksp.setTolerances(rtol=1e-3, atol=1e-10)
# ksp.setType(KSP.Type.NONE) # set to NONE and PC=LU to use direct solver
# pc = PC().create()
# pc.setType(PC.Type.NONE)
# pc.setFactorSolverType(Mat.SolverType.UMFPACK)
# ksp.setPC(pc)
solver.setKSP(ksp)

# solver.setType(SNES.Type.QN)

r = Vec().createSeq(num_nodes)
r.setUp()
J = Mat().createDense((num_nodes, num_nodes))
J.setUp()

x = Vec().createWithArray(u_0)
solver.setFunction(residual_poisson_wrap, f=r, kargs=kargs)
solver.setJacobian(jacobian_poisson_wrap, J=J, kargs=kargs)

solver.setSolution(x)
solver.setFromOptions()

# jacobian_poisson_wrap(solver, x, J, None, star_f.coeffs, boundary_values, gamma)
# jacobian_poisson_wrap(solver, x, J, None, star_f.coeffs, boundary_values, gamma)
# print(J.getDenseArray())
# print(J.getDenseArray())

tic = time.time()

solver.solve()
solver.view()

toc = time.time()

print("Elapsed time = ", toc - tic)

print(solver.getConvergedReason())

u = solver.getSolution()
# residual_poisson_wrap(solver, u, r, star_f.coeffs, boundary_values, gamma)
# print(r.getArray())
u = u.getArray()
# print(u-u_true)

assert np.allclose(u, u_true, atol=1e-2)

log.view()