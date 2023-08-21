# dctkit - Discrete Calculus Toolkit

[![Linting and
testing](https://github.com/alucantonio/dctkit/actions/workflows/tests.yml/badge.svg)](https://github.com/alucantonio/dctkit/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/dctkit/badge/?version=latest)](https://dctkit.readthedocs.io/en/latest/?badge=latest)

`dctkit` implements operators from Algebraic Topology, Discrete Exterior Calculus and
Discrete Differential Geometry to provide a *mathematical language for building discrete physical models*.

Features:
- uses [`jax`](http://github.com/google/jax/) as a backend for numerical computations
- manipulation of simplicial complexes of any dimension: computation of
  boundary/coboundary operators, circumcenters, dual/primal volumes
- manipulation of (primal/dual) cochains: addition, multiplication by scalar, inner
  product, coboundary, Hodge star, codifferential, Laplace-de Rham
- manipulation of vector-valued and tensor-valued cochains: discrete vector and tensor
  fields, sharp operator
- interface to the [`pygmo`](https://github.com/esa/pygmo2) optimization library
- interface to the [`PETSc`](https://petsc.org) non-linear solvers and optimizers
- routines for solving optimal control problems
- implementation of discrete physical models: Dirichlet energy, Poisson, Euler's
  Elastica, isotropic linear elasticity

## Installation

Dependencies should be installed within a `conda` environment. We recommend using
[`mamba`](https://github.com/mamba-org/mamba) since it is much faster than `conda` at
solving the environment and downloading the dependencies. To create a suitable
environment based on the provided `.yaml` file, use the command

```bash
$ mamba env create -f environment.yaml
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
| `python bench_poisson.py scipy cpu`  | 1.553 ± 0.007 |   1.546 |   1.564 | 2.16 ± 0.01 |
| `python bench_poisson.py pygmo cpu`  | 0.719 ± 0.003 |   0.716 |   0.723 |        1.00 |
| `python bench_poisson.py jaxopt cpu` | 2.376 ± 0.055 |   2.309 |   2.451 | 3.30 ± 0.08 |
| `python bench_poisson.py jaxopt gpu` | 5.638 ± 0.040 |   5.590 |   5.699 | 7.84 ± 0.07 |

## Usage

Read the full [documentation](https://dctkit.readthedocs.io/en/latest/) (including API
docs).

*Example*: solving discrete Poisson equation in 1D (variational formulation):

```python
import dctkit as dt
from dctkit import config
from dctkit.mesh import util
from dctkit.dec import cochain as C
from dctkit.math.opt import optctrl as oc
import jax.numpy as jnp
from matplotlib.pyplot import plot

# configure the JAX backend: sets precision and platform (CPU/GPU)
# defaults: float64 precision and CPU platform
# MUST be called before using any function of dctkit
config()

# generate mesh and create SimplicialComplex object
num_nodes = 10
L = 1.
mesh, _ = util.generate_line_mesh(num_nodes, L)
S = util.build_complex_from_mesh(mesh)

# compute Hodge star operator for later use
S.get_hodge_star()

# initial guess for the solution vector (coefficients of a primal 0-chain)
# except for the node 0, where the boundary condition is prescribed
u = jnp.ones(num_nodes-1, dtype=dt.float_dtype)

# source term (primal 0-cochain)
f = C.CochainP0(complex=S, coeffs=jnp.ones(num_nodes))

# discrete Dirichlet energy with source term
def energy(u):
     # zero Dirichlet bc at x=0
     u = jnp.insert(u, 0, 0.)
     # wrap JAX array (when calling minimization routine) into a cochain
     uc = C.CochainP0(complex=S, coeffs=u)
     du = C.coboundary(uc)
     return C.inner_product(du, du)-C.inner_product(uc, f)

# set optimization problem
prb = oc.OptimizationProblem(dim=num_nodes-1, state_dim=num_nodes-1, objfun=energy)
# set additional arguments (empty dictionary) of the objective function,
# other than the state array
prb.set_obj_args({})

x = prb.solve(x0=u)

# add boundary condition
x = jnp.insert(x, 0, 0.)

plot(x)
```