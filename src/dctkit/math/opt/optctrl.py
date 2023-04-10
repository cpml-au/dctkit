import numpy as np
from jax import grad, jit, jacrev, Array
from scipy.optimize import minimize
from typing import Callable
import numpy.typing as npt
from typing import Any, Tuple


class OptimalControlProblem():
    """Class for optimal control problems of the form:

        (u, a) = argmin J(u, a)     s.t.  F(u, a) = 0,

        where J is the objective function and F is the state function.

    Args:
        objfun: objective function to minimize wrt parameters (controls). Its
            arguments must be the state array and the paramters array.
        state_en: energy function to minimize wrt the state to define the current
            state given a set of controls.
        state_dim: number of state variables (dimension of the state array).
    """

    def __init__(self, objfun: Callable, state_en: Callable, state_dim: int,
                 constraint_args: Tuple = (), obj_args: Tuple = ()) -> None:
        self.objfun = jit(objfun)
        self.state_en = jit(state_en)
        # gradient of the state energy wrt state, i.e. state equation
        self.grad_u = jit(grad(self.state_en))
        # length of the state vector
        self.state_dim = state_dim
        # gradient of the objective function wrt the parameters vector
        self.grad_obj = jit(grad(self.obj_fun_wrap))
        # gradient of the state equation wrt the parameters vector
        self.state_eq_grad = jit(jacrev(self.state_eq_wrap))
        self.constraint_args = constraint_args
        self.obj_args = obj_args

    def obj_fun_wrap(self, x: Array | npt.NDArray, *args: Any) -> Array:
        """Wrapper for the objective function.

        Args:
            x: optimization paramters (state + controls).
        Returns:
            value of the objective function.
        """
        u = x[:self.state_dim]
        a = x[self.state_dim:]
        obj = self.objfun(u, a, *args)
        return obj

    def state_eq_wrap(self, x: Array | npt.NDArray, *args: Any) -> Array:
        """Wrapper for the state equation.

        Args:
            x: optimization paramters (state + controls).
        Returns:
            residual of the system of state equations.
        """
        u = x[:self.state_dim]
        a = x[self.state_dim:]
        return self.grad_u(u, a, *args)

    def run(self, u0: npt.NDArray, y0: npt.NDArray, tol: float) \
            -> Tuple[npt.NDArray, npt.NDArray, float]:
        """Solves the optimal control problem by SLSQP.

        Args:
            u0: initial guess for the state.
            a0: initial guess for the parameters (controls).
            tol: controls the tolerance on the objective function value.
        Returns:
            tuple containing the optimal state, the optimal controls and the value of
            the objective function.
        """
        x0 = np.concatenate((u0, y0))
        res = minimize(self.obj_fun_wrap, x0, args=self.obj_args, method="SLSQP",
                       constraints={'type': 'eq', 'fun': self.state_eq_wrap,
                                    'jac': self.state_eq_grad,
                                    'args': self.constraint_args},
                       jac=self.grad_obj, tol=tol, options={'maxiter': 1000})

        u = res.x[:self.state_dim]
        a = res.x[self.state_dim:]
        fval = res.fun
        return u, a, fval
