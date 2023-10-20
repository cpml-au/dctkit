import numpy as np
import dctkit as dt_
from dctkit.physics import burgers as b


def test_burgers(setup_test):
    x_max = 5
    t_max = 2
    dx = 0.025
    dt = 0.001
    num_x_points = int(x_max/dx)
    num_t_points = int(t_max/dt)
    x = np.linspace(0, x_max, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # boundary and initial conditions
    u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * x_max)**2)
    nodes_BC = {'left': np.zeros(num_t_points), 'right': np.zeros(num_t_points)}

    # viscosity coefficient
    epsilon = 0.1

    # define u_FDM
    def FDM_run(u, epsilon, scheme="parabolic"):
        for t in range(num_t_points - 1):
            diffusion = epsilon * (u[2:, t] - 2 * u[1:-1, t] + u[:-2, t]) / dx**2
            if scheme == "parabolic":
                flux = (u[2:, t]**2 - u[:-2, t]**2) / (4*dx)
            elif scheme == "upwind":
                flux = (u[1:-1, t]**2 - u[:-2, t]**2) / (2*dx)
            u[1:-1, t+1] = u[1:-1, t] + dt * (diffusion - flux)
        return u

    u_FDM_par = np.zeros([num_x_points-1, num_t_points], dtype=dt_.float_dtype)
    u_FDM_par[:, 0] = u_0.copy()
    u_FDM_par[0, :] = nodes_BC['left']
    u_FDM_par[-1, :] = nodes_BC['right']
    u_FDM_up = u_FDM_par.copy()
    u_FDM_par = FDM_run(u_FDM_par, epsilon, "parabolic")
    u_FDM_up = FDM_run(u_FDM_up, 0, "upwind")

    prb_par = b.Burgers(x_max, t_max, dx, dt, u_0, nodes_BC, epsilon)
    prb_up = b.Burgers(x_max, t_max, dx, dt, u_0, nodes_BC, 0)
    prb_par.run(scheme="parabolic")
    prb_up.run(scheme="upwind")

    total_err_par = np.mean(np.linalg.norm(u_FDM_par - prb_par.u, axis=0)**2)
    total_err_up = np.mean(np.linalg.norm(u_FDM_up - prb_up.u, axis=0)**2)

    assert total_err_par < 5e-2
    assert total_err_up < 5e-2
