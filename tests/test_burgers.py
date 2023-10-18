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

    # boundary and initial conditions
    u_0 = 2 * np.exp(-2 * (x - 0.5 * x_max)**2)
    nodes_BC = {'left': np.zeros(num_t_points), 'right': np.zeros(num_t_points)}

    # viscosity coefficient
    epsilon = 0.1

    # define u_FDM
    def FDM_run(u):
        for t in range(num_t_points - 1):
            diffusion = epsilon * (u[2:, t] - 2 * u[1:-1, t] + u[:-2, t]) / dx**2
            flux = (u[2:, t]**2 - u[:-2, t]**2) / (4*dx)
            u[1:-1, t+1] = u[1:-1, t] + dt * (diffusion - flux)
        return u

    u_FDM = np.zeros([num_x_points, num_t_points], dtype=dt_.float_dtype)
    u_FDM[:, 0] = u_0.copy()
    u_FDM[0, :] = nodes_BC['left']
    u_FDM[-1, :] = nodes_BC['right']
    u_FDM = FDM_run(u_FDM)

    prb = b.Burgers(x_max, t_max, dx, dt, u_0, nodes_BC, epsilon)
    prb.run()

    errors = np.array([np.linalg.norm(u_FDM[:, t+1] - prb.u[:, t+1])
                       for t in range(num_t_points-1)])

    assert np.mean(errors) < 5e-2
