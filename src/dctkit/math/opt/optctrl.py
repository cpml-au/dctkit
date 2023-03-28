import numpy as np
import jax.numpy as jnp
from jax import grad, jit, jacrev
from scipy.optimize import minimize


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

    def __init__(self, objfun: callable, state_en: callable, state_dim: int) -> None:
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

    def obj_fun_wrap(self, x: jnp.array) -> jnp.array:
        """Wrapper for the objective function.

        Args:
            x: optimization paramters (state + controls).
        Returns:
            value of the objective function.
        """
        u = x[:self.state_dim]
        a = x[self.state_dim:]
        obj = self.objfun(u, a)
        return obj

    def state_eq_wrap(self, x: jnp.array) -> jnp.array:
        """Wrapper for the state equation.

        Args:
            x: optimization paramters (state + controls).
        Returns:
            residual of the system of state equations.
        """
        u = x[:self.state_dim]
        a = x[self.state_dim:]
        return self.grad_u(u, a)

    def run(self, u0: np.array, y0: np.array, tol: float) -> (np.array, np.array, float):
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
        res = minimize(self.obj_fun_wrap, x0, method="SLSQP", constraints={
                       'type': 'eq', 'fun': self.state_eq_wrap, 'jac': self.state_eq_grad}, jac=self.grad_obj, tol=tol)
        print(res)
        u = res.x[:self.state_dim]
        a = res.x[self.state_dim:]
        fval = res.fun
        return u, a, fval
