import numpy as np
from jax import grad, jit, jacrev, Array
from typing import Callable
import numpy.typing as npt
from typing import List, Dict
import pygmo as pg
from typeguard import check_type


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
                                   np.float64 | npt.NDArray | Array] = {}) -> None:
        self.dim = dim
        self.state_dim = state_dim
        self.obj = jit(objfun)
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
        # gradient of the objective function wrt parameters array
        self.grad_obj = jit(grad(objfun))
        self.last_opt_result = -1

    def set_obj_args(self, args: dict) -> None:
        """Sets the additional arguments to be passed to the objective function."""
        check_type(args, Dict[str, float | np.float32 | np.float64
                              | npt.NDArray | Array])
        self.obj_args = args

    def get_nec(self) -> int:
        """Returns the number of equality constraints: for a constrained problem, it
        is equal to the number of state variables, otherwise it is zero.
        """
        if self.constr_problem:
            return self.state_dim
        else:
            return 0

    def fitness(self, x: npt.NDArray | Array) -> npt.NDArray | List[float]:
        fit = self.obj(x, **self.obj_args)
        check_type(fit, float | np.float32 | np.float64 | npt.NDArray | Array)
        if self.constr_problem:
            constr_res = self.constr(x, **self.constr_args)
            return np.concatenate(([fit], constr_res))
        else:
            return [fit]

    def gradient(self, x: npt.NDArray | Array) -> npt.NDArray | Array:
        grad = self.grad_obj(x, **self.obj_args)
        check_type(grad, float | np.float32 | np.float64 | npt.NDArray | Array)
        if self.constr_problem:
            constr_jac = self.constr_grad(x, **self.constr_args)
            # first dim components are grad of object wrt parameters, then grad of
            # constraint equations wrt parameters.
            return np.concatenate((grad, np.ravel(constr_jac)))
        else:
            return grad

    def get_bounds(self):
        return ([-100]*self.dim, [100]*self.dim)

    def get_name(self) -> str:
        """Returns the name of the optimization problem. Override this method to set
        another name.
        """
        return "Optimization problem"

    def run(self, x0: npt.NDArray, algo: str = "tnewton", ftol_abs: float = 1e-5,
            ftol_rel: float = 1e-5, maxeval: int = 500) -> npt.NDArray:
        prb = pg.problem(self)

        if self.constr_problem:
            algo = "slsqp"
        algo = pg.algorithm(pg.nlopt(solver=algo))
        algo.extract(pg.nlopt).ftol_abs = ftol_abs  # type: ignore
        algo.extract(pg.nlopt).ftol_rel = ftol_rel  # type: ignore
        algo.extract(pg.nlopt).maxeval = maxeval  # type: ignore
        pop = pg.population(prb)
        pop.push_back(x0)
        # algo.set_verbosity(1)
        pop = algo.evolve(pop)  # type: ignore
        self.last_opt_result = algo.extract(  # type: ignore
            pg.nlopt).get_last_opt_result()
        u = pop.champion_x
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
