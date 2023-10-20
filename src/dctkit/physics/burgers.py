import dctkit as dt_
from dctkit.mesh import util
import numpy as np
from dctkit.dec import cochain as C
from dctkit.dec.vector import flat_PDD as flat
import numpy.typing as npt
from typing import Dict


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

    def __init__(self, x_max: float, t_max: float, dx: float, dt: float,
                 u_0: npt.NDArray, nodes_BC: Dict, epsilon: float):
        self.x_max = x_max
        self.t_max = t_max
        self.dx = dx
        self.dt = dt
        self.u_0 = u_0
        self.nodes_BC = nodes_BC
        self.epsilon = epsilon
        self.num_x_points = int(self.x_max/self.dx)
        self.num_t_points = int(self.t_max/self.dt)
        # define complex
        self.__get_burgers_mesh()
        # initialize u with boundary and initial conditions
        self.set_u_BC_IC()

    def __get_burgers_mesh(self):
        """Define simplicial complex."""
        mesh, _ = util.generate_line_mesh(self.num_x_points, self.x_max)
        self.S = util.build_complex_from_mesh(mesh)
        self.S.get_hodge_star()
        self.S.get_flat_PDP_weights()

    def set_u_BC_IC(self):
        """Set boundary and initial conditions to u."""
        self.u = np.zeros((self.num_x_points - 1, self.num_t_points),
                          dtype=dt_.float_dtype)
        # set initial conditions
        self.u[:, 0] = self.u_0
        self.u[0, :] = self.nodes_BC['left']
        self.u[-1, :] = self.nodes_BC['right']

    def run(self, scheme: str = "parabolic"):
        """Main run to solve Burgers' equation with DEC.

        Args:
            scheme: discretization scheme used.
        """
        for t in range(self.num_t_points - 1):
            u_coch = C.CochainD0(self.S, self.u[:, t])
            dissipation = C.scalar_mul(C.star(C.coboundary(u_coch)), self.epsilon)
            if scheme == "upwind":
                flat_u = flat(u_coch, scheme)
                flux = C.scalar_mul(C.square(C.star(flat_u)), -1/2)
            elif scheme == "parabolic":
                u_sq = C.scalar_mul(C.square(u_coch), -1/2)
                flux = C.star(flat(u_sq, scheme))
            total_flux = C.add(flux, dissipation)
            balance = C.star(C.coboundary(total_flux))
            self.u[1:-1, t+1] = self.u[1:-1, t] + self.dt*balance.coeffs[1:-1]
