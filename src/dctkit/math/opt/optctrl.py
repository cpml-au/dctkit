import numpy as np
from jax import grad, jit, jacrev, value_and_grad, Array
from typing import Callable
import numpy.typing as npt
from typing import List, Dict
import pygmo as pg
from typeguard import check_type
from petsc4py import PETSc, init
from petsc4py.PETSc import Vec


class OptimizationProblem():
    """Class for (constrained) optimization problems.

    Args:
        dim: dimension of the parameters array (state + controls). state_dim: dimension
            of the state array.
        objfun: objective function. Its arguments must be the parameters array (state +
            constrols) and some extra arguments to be set using the method
            `set_obj_args'.
    """

    def __init__(self, dim: int, state_dim: int,
                 objfun: Callable[..., float | npt.NDArray | Array |
                                  np.float32 | np.float64],
                 constrfun: Callable[..., float | npt.NDArray | Array |
                                     np.float32 | np.float64] | None = None,
                 constr_args: Dict[str, float | np.float32 |
                                   np.float64 | npt.NDArray | Array] = {},
                 solver_lib: str = 'pygmo'):

        self.dim = dim
        self.state_dim = state_dim

        self.__register_obj_and_grad_fn(objfun, constrfun, constr_args)

        if solver_lib == "petsc":
            self.solver = PETScSolver(self)
        else:
            self.solver = PygmoSolver(self)

        self.last_opt_result = -1

    def __register_obj_and_grad_fn(self, objfun, constrfun, constr_args):
        self.obj = jit(objfun)
        # gradient of the objective function wrt parameters array
        self.grad_obj = jit(grad(objfun))

        self.objandgrad = jit(value_and_grad(objfun))

        self.constr_problem = False
        # constrained optimization problem
        if constrfun is not None:
            self.constr = jit(constrfun)
            # jacobian of the constraint equations wrt the parameters array
            self.constr_grad = jit(jacrev(constrfun))
            # TODO: check according to dt.float_dtype instead of np.float32 or 64
            check_type(constr_args, Dict[str, float | np.float32 | np.float64 |
                                         npt.NDArray | Array])
            self.constr_args = constr_args
            self.constr_problem = True

    def set_obj_args(self, args: dict) -> None:
        """Sets the additional arguments to be passed to the objective function."""
        self.solver.set_obj_args(args)

    def solve(self, x0: npt.NDArray, algo: str = "tnewton", ftol_abs: float = 1e-5,
              ftol_rel: float = 1e-5, gatol=None, grtol=None, gttol=None,
              maxeval: int = 500, verbose: bool = False) -> npt.NDArray:
        
        # FIXME: these parameters should be set through a separate function of the
        # solver

        kwargs = {"algo": algo, "ftol_abs": ftol_abs, "ftol_rel": ftol_rel,
                  "gatol": gatol, "grtol": grtol, "gttol": gttol, "maxeval": maxeval,
                  "verbose": verbose}

        u = self.solver.run(x0, **kwargs)

        check_type(u, npt.NDArray)
        return u


class OptimalControlProblem(OptimizationProblem):
    """Class for optimal control problems of the form:

        x = argmin_a J(x)     s.t.  F(x) = 0,

        where x = (u,a), u is the state, a are the controls, J is the objective function
        and F is the state function.

    Args:
        objfun: objective function to minimize wrt controls. Its
            argument must be the parameter (state + controls) array. Additional
            arguments must be specified via the parameters `obj_args'.
        statefun: function computing the residual vector of the state equations. Its
            arguments must be the parameters array and other keyword arguments specified
            via the parameter `constraint_args'.
        state_dim: number of state variables (dimension of the state array).
        nparams: number of optimization parameters (state + controls).
        constraint_args: extra keyword arguments for the state function.
        obj_args: extra keyword arguments for the objective function.
    """

    def __init__(self, objfun: Callable[..., np.float32 | np.float64 |
                                        npt.NDArray | Array],
                 statefun: Callable[..., np.float32 | np.float64 | npt.NDArray | Array],
                 state_dim: int, nparams: int,
                 constraint_args: Dict[str, float | np.float32 |
                                       np.float64 | npt.NDArray | Array] = {},
                 obj_args: Dict[str, float | np.float32 |
                                np.float64 | npt.NDArray | Array] = {}) -> None:

        super().__init__(dim=nparams, state_dim=state_dim, objfun=objfun,
                         constrfun=statefun, constr_args=constraint_args)
        super().set_obj_args(obj_args)


class OptimizationSolver():
    def __init__(self, prb: OptimizationProblem):
        self.prb = prb

    def set_obj_args(self, args: Dict):
        """Sets the additional keyword arguments to be passed to the objective
        function."""
        raise NotImplementedError

    def run(self, x0: npt.NDArray, **kwargs: Dict) -> npt.NDArray | Array:
        # FIXME: remove kwargs from run method and implement separate methods for
        # solver settings
        raise NotImplementedError


class PETScSolver(OptimizationSolver):
    def __init__(self, prb: OptimizationProblem):
        super().__init__(prb)
        init()
        # create default solver and settings
        self.tao = PETSc.TAO().create()
        self.tao.setType(PETSc.TAO.Type.LMVM)  # Specify the solver type
        # create variable to store the gradient of the objective function
        self.g = PETSc.Vec().createSeq(self.prb.dim)

    def set_obj_args(self, args: dict) -> None:
        check_type(args, Dict[str, float | np.float32 | np.float64
                              | npt.NDArray | Array])
        self.tao.setObjectiveGradient(
            self.objective_and_gradient, self.g, kargs=args)

    def objective_and_gradient(self, tao, x: Vec, g: Vec, **kwargs: Dict):
        """PETSc-compatible wrapper for the function that returns the objective value
        and the gradient."""
        fval, grad = self.prb.objandgrad(x.getArray(), **kwargs)
        g.setArray(grad)
        return fval

    def run(self, x0: npt.NDArray, **kwargs: Dict) -> npt.NDArray:
        maxeval = kwargs["maxeval"]
        gatol = kwargs["gatol"]
        grtol = kwargs["grtol"]
        gttol = kwargs["gttol"]
        verbose = kwargs["verbose"]
        x = PETSc.Vec().createWithArray(x0)
        self.tao.setSolution(x)
        self.tao.setMaximumIterations(maxeval)
        self.tao.setTolerances(gatol=gatol, grtol=grtol, gttol=gttol)
        self.tao.setFromOptions()  # Set options for the solver
        self.tao.solve()
        if verbose:
            self.tao.view()
        u = self.tao.getSolution().getArray()
        # objective_value = self.tao.getObjectiveValue()
        return u


class PygmoSolver(OptimizationSolver):
    def __init__(self, prb: OptimizationProblem):
        super().__init__(prb)
        self.prb.obj_args = {}

    def set_obj_args(self, args: Dict):
        self.prb.obj_args = args

    def get_nec(self) -> int:
        """Returns the number of equality constraints: for a constrained problem, it
        is equal to the number of state variables, otherwise it is zero.
        """
        if self.prb.constr_problem:
            return self.prb.state_dim
        else:
            return 0

    def fitness(self, x: npt.NDArray | Array) -> npt.NDArray | List[float]:
        fit = self.prb.obj(x, **self.prb.obj_args)
        check_type(fit, float | np.float32 | np.float64 | npt.NDArray | Array)
        if self.prb.constr_problem:
            constr_res = self.prb.constr(x, **self.prb.constr_args)
            return np.concatenate(([fit], constr_res))
        else:
            return [fit]

    def gradient(self, x: npt.NDArray | Array) -> npt.NDArray | Array:
        grad = self.prb.grad_obj(x, **self.prb.obj_args)
        check_type(grad, float | np.float32 | np.float64 | npt.NDArray | Array)
        if self.prb.constr_problem:
            constr_jac = self.prb.constr_grad(x, **self.prb.constr_args)
            # first dim components are grad of object wrt parameters, then grad of
            # constraint equations wrt parameters.
            return np.concatenate((grad, np.ravel(constr_jac)))
        else:
            return grad

    def get_bounds(self):
        return ([-100]*self.prb.dim, [100]*self.prb.dim)

    def get_name(self) -> str:
        """Returns the name of the optimization problem. Override this method to set
        another name.
        """
        return "Optimization problem"

    def run(self, x0: npt.NDArray, **kwargs) -> npt.NDArray:
        algo = kwargs["algo"]
        ftol_abs = kwargs["ftol_abs"]
        ftol_rel = kwargs["ftol_rel"]
        maxeval = kwargs["maxeval"]
        verbose = kwargs["verbose"]

        pygmo_prb = pg.problem(self)

        if self.prb.constr_problem:
            algo = "slsqp"
        algo = pg.algorithm(pg.nlopt(solver=algo))
        algo.extract(pg.nlopt).ftol_abs = ftol_abs  # type: ignore
        algo.extract(pg.nlopt).ftol_rel = ftol_rel  # type: ignore
        algo.extract(pg.nlopt).maxeval = maxeval  # type: ignore
        pop = pg.population(pygmo_prb)
        pop.push_back(x0)
        algo.set_verbosity(verbose)
        pop = algo.evolve(pop)  # type: ignore
        self.prb.last_opt_result = algo.extract(  # type: ignore
            pg.nlopt).get_last_opt_result()
        u = pop.champion_x
        return u
