# dctkit - Discrete Calculus Toolkit

`dctkit` implements operators from Algebraic Topology, Discrete Exterior Calculus and
Discrete Differential Geometry to provide a mathematical language for building discrete
physical models.

Features:
- supports `numpy` and [`jax`](http://github.com/google/jax/) backends for numerical computations
- manipulation of simplicial complexes of any dimension: computation of boundary/coboundary operators, circumcenters, dual/primal volumes
- manipulation of (primal/dual) cochains: addition, multiplication by scalar, inner product, coboundary, Hodge star, codifferential, Laplace-de Rham
- interface for solving optimal control problems (using `SciPy` constrained optimization
  routines)
- implements the discrete Dirichlet energy and the discrete Poisson model
- discrete Euler's Elastica model
- benchmarks using different optimization packages

## Installation

Clone the git repository and launch the following command

```bash
$ pip install -e .
```

to install a development version of the `dctkit` library.

Running the tests:

```bash
$ tox
```

Generating the docs:

```bash
$ tox -e docs
```

Running the benchmarks:

```bash
$ sh ./bench/run_bench
```
The ASCIIdoc file `bench_results.adoc` will be generated. To view the results in HTML,
convert the `.adoc` file using the command `asciidoc bench_results.adoc`.

## Usage

Solving discrete Poisson equation in 1D (variational formulation):

```python
import dctkit as dt
from dctkit import config, FloatDtype, IntDtype, Backend, Platform
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
from matplotlib.pyplot import plot

# set backend for computations, precision and platform (CPU/GPU)
# MUST be called before using any function of dctkit
config(FloatDtype.float32, IntDtype.int32, Backend.jax, Platform.cpu)

# generate mesh and create SimplicialComplex object
num_nodes = 10
L = 1.
S_1, x = util.generate_1_D_mesh(num_nodes, L)
S = simplex.SimplicialComplex(S_1, x, is_well_centered=True)
# perform some computations and cache results for later use
S.get_circumcenters()
S.get_primal_volumes()
S.get_dual_volumes()
S.get_hodge_star()

# initial guess for the solution vector (coefficients of a primal 0-chain)
u = jnp.ones(num_nodes, dtype=dt.float_dtype)

# source term (primal 0-cochain)
f = C.CochainP0(complex=S, coeffs=jnp.ones(num_nodes))

# discrete Dirichlet energy with source term
def energy(u):
     # wrap np.array (when called by scipy's minimize) into a cochain
     uc = C.CochainP0(complex=S, coeffs=u)
     du = C.coboundary(uc)
     return C.inner_product(du, du)-C.inner_product(uc, f)

# compute gradient of the energy using JAX's autodiff
graden = jit(grad(energy))

# zero Dirichlet bc at x=0
cons = {'type': 'eq', 'fun': lambda x: x[0]}

# constrained minimization of the energy
res = minimize(fun=energy, x0=u, constraints=cons, jac=graden)

print(res)
plot(res.x)
```