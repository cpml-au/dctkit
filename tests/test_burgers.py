import numpy as np
import dctkit as dt_
from dctkit.physics import burgers as b


def test_burgers(setup_test):
    # SPACE PARAMS
    L = 5
    L_norm = 1
    # spatial resolution
    dx = 0.05
    dx_norm = dx/L
    #  Number of spatial grid points
    num_x_points = int(L / dx)
    num_x_points_norm = num_x_points

    # vector containing spatial points
    x = np.linspace(0, L, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # initial velocity
    u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * L)**2)
    umax = np.max(u_0)

    # TIME PARAMS
    T = 2
    T_norm = T*umax/L
    # temporal resolution
    dt = 0.01
    dt_norm = dt*umax/L
    # number of temporal grid points
    num_t_points_norm = int(T_norm / dt_norm)

    # Viscosity
    epsilon = 0.05
    epsilon_norm = epsilon/(L*umax)

    nodes_BC = {'left': np.zeros(num_t_points_norm),
                'right': np.zeros(num_t_points_norm)}

    # define u_FDM
    def FDM_run(u, epsilon, scheme="parabolic"):
        for t in range(num_t_points_norm - 1):
            diffusion = epsilon * (u[2:, t] - 2 * u[1:-1, t] + u[:-2, t]) / dx_norm**2
            if scheme == "parabolic":
                flux = (u[2:, t]**2 - u[:-2, t]**2) / (4*dx_norm)
            elif scheme == "upwind":
                flux = (u[1:-1, t]**2 - u[:-2, t]**2) / (2*dx_norm)
            u[1:-1, t+1] = u[1:-1, t] + dt_norm * (diffusion - flux)
        return u

    u_FDM_par = np.zeros([num_x_points_norm-1, num_t_points_norm],
                         dtype=dt_.float_dtype)
    u_FDM_par[:, 0] = u_0.copy()/umax
    u_FDM_par[0, :] = nodes_BC['left']
    u_FDM_par[-1, :] = nodes_BC['right']
    u_FDM_up = u_FDM_par.copy()
    u_FDM_par = FDM_run(u_FDM_par, epsilon_norm, "parabolic")
    u_FDM_up = FDM_run(u_FDM_up, 0, "upwind")

    prb_par = b.Burgers(L_norm, T_norm, dx_norm, dt_norm,
                        u_0/umax, nodes_BC, epsilon_norm)
    prb_up = b.Burgers(L_norm, T_norm, dx_norm, dt_norm, u_0/umax, nodes_BC, 0)
    prb_par.run(scheme="parabolic")
    prb_up.run(scheme="upwind")

    total_err_par = np.mean(np.linalg.norm(u_FDM_par - prb_par.u, axis=0)**2)
    total_err_up = np.mean(np.linalg.norm(u_FDM_up - prb_up.u, axis=0)**2)

    assert total_err_par < 1e-2
    assert total_err_up < 1e-2
