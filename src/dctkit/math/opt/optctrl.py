import numpy as np
from jax import grad, jit, jacrev, Array
# from scipy.optimize import minimize
from scipy import optimize
from typing import Callable
import numpy.typing as npt
from typing import Any, Tuple
import pygmo as pg


class OptimizationProblem():
    """Class for (constrained) optimization problems.

    Args:
        dim: dimension of the parameters array (state + controls).
        objfun: objective function. Its arguments must be the parameters array (state +
        constrols) and some extra arguments to be set using the method `set_obj_args'.
    """

    def __init__(self, dim: int, state_dim: int, objfun: Callable, constrfun: Callable | None = None, constr_args: dict = {}) -> None:
        self.dim = dim
        self.state_dim = state_dim
        self.obj = jit(objfun)
        self.constr_problem = False
        # constrained optimization problem
        if constrfun is not None:
            self.constr = jit(constrfun)
            # jacobian of the constraint equations wrt the parameters array
            self.constr_grad = jit(jacrev(constrfun))
            self.constr_args = constr_args
            self.constr_problem = True
        # gradient of the objective function wrt parameters array
        self.grad_obj = jit(grad(objfun))

    def set_obj_args(self, args: dict) -> None:
        self.obj_args = args

    def get_nec(self):
        return self.dim - self.state_dim

    def fitness(self, x):
        fit = self.obj(x, **self.obj_args)
        if self.constr_problem:
            constr_res = self.constr(x, **self.constr_args)
            return np.concatenate(([fit], constr_res))
        else:
            return [fit]

    def gradient(self, x):
        grad = self.grad_obj(x, **self.obj_args)
        if self.constr_problem:
            constr_jac = self.constr_grad(x, **self.constr_args)
            # first dim components are grad of object wrt parameters, then grad of
            # constraint equations wrt parameters.
            return np.concatenate((grad, np.ravel(constr_jac)))
        else:
            return grad

    def get_bounds(self):
        return ([-100]*self.dim, [100]*self.dim)

    def get_name(self):
        return "Optimization problem"

    def run(self, x0: npt.NDArray, algo: str = "tnewton", ftol_abs: float = 1e-5,
            ftol_rel: float = 1e-5) -> npt.NDArray:
        prb = pg.problem(self)
        print(prb)
        algo = pg.algorithm(pg.nlopt(solver=algo))
        algo.extract(pg.nlopt).ftol_abs = ftol_abs  # type: ignore
        algo.extract(pg.nlopt).ftol_rel = ftol_rel  # type: ignore
        pop = pg.population(prb)
        pop.push_back(x0)
        # print(algo)
        # algo.set_verbosity(1)
        pop = algo.evolve(pop)  # type: ignore
        u = pop.champion_x
        return u


class OptimalControlProblem(OptimizationProblem):
    """Class for optimal control problems of the form:

        (u, a) = argmin_a J(u, a)     s.t.  F(u, a) = 0,

        where J is the objective function and F is the state function.

    Args:
        objfun: objective function to minimize wrt controls. Its
            argument must be the parameter (state + controls) array. Additional
            arguments must be specified via the parameters `obj_args'.
        statefun: function computing the residual vector of the state equations. Its
            arguments must be the parameters array and other keyword arguments specified
            via the parameter `constraint_args'.
        state_dim: number of state variables (dimension of the state array).
        nparams: number of parameters (state + controls).
        constraint_args: extra keyword arguments for the state function.
        obj_args: extra keyword arguments for the objective function.
    """

    def __init__(self, objfun: Callable, statefun: Callable,
                 state_dim: int, nparams: int, constraint_args: dict = {},
                 obj_args: dict = {}) -> None:

        super().__init__(dim=nparams, state_dim=state_dim, objfun=objfun,
                         constrfun=statefun, constr_args=constraint_args)
        super().set_obj_args(obj_args)

    # def obj_fun_wrap(self, x: Array | npt.NDArray, *args: Any) -> Array:
    #     """Wrapper for the objective function.

    #     Args:
    #         x: optimization paramters (state + controls).
    #     Returns:
    #         value of the objective function.
    #     """
    #     u = x[:self.state_dim]
    #     a = x[self.state_dim:]
    #     obj = self.objfun(u, a, *args)
    #     return obj

    # def state_eq_wrap(self, x: Array | npt.NDArray, *args: Any) -> Array:
    #     """Wrapper for the state equation.

    #     Args:
    #         x: optimization paramters (state + controls).
    #     Returns:
    #         residual of the system of state equations.
    #     """
    #     u = x[:self.state_dim]
    #     a = x[self.state_dim:]
    #     return self.grad_u(u, a, *args)

    # def run(self, u0: npt.NDArray, y0: npt.NDArray, tol: float) \
    #         -> Tuple[npt.NDArray, npt.NDArray, float]:
    #     """Solves the optimal control problem by SLSQP.

    #     Args:
    #         u0: initial guess for the state.
    #         a0: initial guess for the parameters (controls).
    #         tol: controls the tolerance on the objective function value.
    #     Returns:
    #         tuple containing the optimal state, the optimal controls and the value of
    #         the objective function.
    #     """
    #     x0 = np.concatenate((u0, y0))
    #     res = optimize.minimize(self.obj_fun_wrap, x0, args=self.obj_args,
    #                             method="SLSQP",
    #                             constraints={'type': 'eq', 'fun': self.state_eq_wrap,
    #                                          'jac': self.state_eq_grad,
    #                                          'args': self.constraint_args},
    #                             jac=self.grad_obj, tol=tol, options={'maxiter': 1000})

    #     u = res.x[:self.state_dim]
    #     a = res.x[self.state_dim:]
    #     fval = res.fun
    #     return u, a, fval
