# dctkit - Discrete Calculus Toolkit

[![Linting and
testing](https://github.com/alucantonio/dctkit/actions/workflows/tests.yml/badge.svg)](https://github.com/alucantonio/dctkit/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/dctkit/badge/?version=latest)](https://dctkit.readthedocs.io/en/latest/?badge=latest)

`dctkit` implements operators from Algebraic Topology, Discrete Exterior Calculus and
Discrete Differential Geometry to provide a *mathematical language for building discrete physical models*.

Features:
- supports `numpy` and [`jax`](http://github.com/google/jax/) backends for numerical computations
- manipulation of simplicial complexes of any dimension: computation of boundary/coboundary operators, circumcenters, dual/primal volumes
- manipulation of (primal/dual) cochains: addition, multiplication by scalar, inner product, coboundary, Hodge star, codifferential, Laplace-de Rham
- interface to different optimization packages: [`SciPy`](https://github.com/scipy/scipy), [`jaxopt`](http://github.com/google/jaxopt), and [`pygmo`](https://github.com/esa/pygmo2)
- interface for solving optimal control problems
- implements the discrete Dirichlet energy and the discrete Poisson model
- discrete Euler's Elastica model

## Installation

Dependencies should be installed within a `conda` environment. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ conda env create -f environment.yaml
```

Otherwise, update an existing environment using the same `.yaml` file.

After activating the environment, clone the git repository and launch the following command

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
The Markdown file `bench_results.md` will be generated containing the results.

*Reference performance (HP Z2 Workstation G9 - 12th Gen Intel i9-12900K (24) @ 5.200GHz - NVIDIA RTX A4000 - 64GB RAM - ArchLinux kernel v6.2.9)*

| Command                              |      Mean [s] | Min [s] | Max [s] |    Relative |
| :----------------------------------- | ------------: | ------: | ------: | ----------: |
| `python bench_poisson.py scipy cpu`  | 1.329 ± 0.028 |   1.307 |   1.379 | 1.88 ± 0.05 |
| `python bench_poisson.py pygmo cpu`  | 0.708 ± 0.011 |   0.692 |   0.722 |        1.00 |
| `python bench_poisson.py nlopt cpu`  | 0.760 ± 0.008 |   0.750 |   0.769 | 1.07 ± 0.02 |
| `python bench_poisson.py jaxopt cpu` | 2.291 ± 0.015 |   2.277 |   2.308 | 3.24 ± 0.06 |
| `python bench_poisson.py jaxopt gpu` | 5.572 ± 0.033 |   5.527 |   5.612 | 7.87 ± 0.13 |

## Usage

Read the full [documentation](https://dctkit.readthedocs.io/en/latest/) (including API
docs).

*Example*: solving discrete Poisson equation in 1D (variational formulation):

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