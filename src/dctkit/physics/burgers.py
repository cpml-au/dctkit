import dctkit as dt_
import numpy as np
from dctkit.dec import cochain as C
from dctkit.dec.vector import flat_PDD as flat
import numpy.typing as npt
from typing import Dict
import math


class Burgers():
    """Burgers' problem class.

    Args:
        x_max: maximum x.
        t_max: maximum t.
        dx: spatial resolution.
        dt: temporal resolution.
        u_0: initial condition on u.
        nodes_BC: boundary conditions on u.
        epsilon: viscosity.
    """

    def __init__(self, S, t_max: float, dt: float, u_0: npt.NDArray,
                 nodes_BC: Dict, epsilon: float):
        self.S = S
        self.t_max = t_max
        self.dt = dt
        self.u_0 = u_0
        self.nodes_BC = nodes_BC
        self.epsilon = epsilon
        # simple trick to round up
        self.num_t_points = int(math.ceil(self.t_max/self.dt))
        self.u_dot = np.zeros(
            (self.S.num_nodes-1, self.num_t_points), dtype=dt_.float_dtype)
        self.set_u_BC_IC()

    def set_u_BC_IC(self):
        """Set boundary and initial conditions to u."""
        self.u = np.zeros((self.S.num_nodes - 1, self.num_t_points),
                          dtype=dt_.float_dtype)
        # set initial conditions
        self.u[:, 0] = self.u_0
        self.u[0, :] = self.nodes_BC['left']
        self.u[-1, :] = self.nodes_BC['right']

    def compute_time_balance(self, t_idx: float, scheme: str = "parabolic"):
        u_coch = C.CochainD0(self.S, self.u[:, t_idx])
        dissipation = C.scalar_mul(C.star(C.coboundary(u_coch)), self.epsilon)
        if scheme == "upwind":
            flat_u = flat(u_coch, scheme)
            flux = C.scalar_mul(C.square(C.star(flat_u)), -1/2)
        elif scheme == "parabolic":
            u_sq = C.scalar_mul(C.square(u_coch), -1/2)
            flux = C.star(flat(u_sq, scheme))
        total_flux = C.add(flux, dissipation)
        balance = C.star(C.coboundary(total_flux))
        return balance

    def run(self, scheme: str = "parabolic"):
        """Main run to solve Burgers' equation with DEC.

        Args:
            scheme: discretization scheme used.
        """
        for t in range(self.num_t_points - 1):
            balance = self.compute_time_balance(t, scheme)
            self.u_dot[1:-1, t] = balance.coeffs[1:-1]
            self.u[1:-1, t+1] = self.u[1:-1, t] + self.dt*self.u_dot[1:-1, t]

        self.u_dot[1:-1, -1] = self.compute_time_balance(-1, scheme).coeffs[1:-1]
