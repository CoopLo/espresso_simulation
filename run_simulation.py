import numpy as np
from numba import jit
from generate_bed import load_bed
from tqdm import tqdm


#@jit
def x_momentum(u_grid, v_grid, dx, dy, dt, nu):
    u_star_grid = np.zeros(u_grid.shape)

    for i in range(1, u_grid.shape[0]-1):
        for j in range(1, u_grid.shape[1]-1):

            # Diffusion terms
            diff_x = 4*nu/(dx**2) * (u_grid[i+1][j] - u_grid[i][j] + u_grid[i-1][j])
            diff_y = 4*nu/(dy**2) * (u_grid[i][j+1] - u_grid[i][j] + u_grid[i][j-1])

            # duu/dx
            duu = 1/dx * (((u_grid[i+1][j]+u_grid[i][j])/2)**2 - \
                          ((u_grid[i][j] + u_grid[i-1][j])/2)**2)

            # dvu/dy
            dvu = 1/dy * (((u_grid[i+1][j]+u_grid[i][j])/2)*((v_grid[i+1][j]+v_grid[i][j])/2) - \
                        ((u_grid[i][j]+u_grid[i-1][j])/2)*((v_grid[i+1][j-1]+v_grid[i][j-1])/2))

            # Update grid
            u_star_grid[i][j] = u_grid[i][j] + dt*(diff_x + diff_y - duu - dvu)

    return u_star_grid


#@jit
def y_momentum(u_grid, v_grid, dx, dy, dt, nu):
    v_star_grid = np.zeros(v_grid.shape)

    for i in range(1, v_grid.shape[0]-1):
        for j in range(1, v_grid.shape[1]-1):

            # Diffvsion terms
            diff_x = 4*nu/(dx**2) * (v_grid[i+1][j] - v_grid[i][j] + v_grid[i-1][j])
            diff_y = 4*nu/(dy**2) * (v_grid[i][j+1] - v_grid[i][j] + v_grid[i][j-1])

            # dvv/dx
            dvv = 1/dx * (((v_grid[i][j+1]+v_grid[i][j])/2)**2 - \
                          ((v_grid[i][j]+v_grid[i][j-1])/2)**2)

            # duv/dy TODO: Check this. It may be incorrect.
            duv = 1/dy * (((u_grid[i][j+1]+u_grid[i][j])/2)*((v_grid[i+1][j]+v_grid[i][j])/2) - \
                        ((u_grid[i-1][j]+u_grid[i-1][j-1])/2)*((v_grid[i][j]+v_grid[i-1][j])/2))

            # Update grid
            v_star_grid[i][j] = v_grid[i][j] + dt*(diff_x + diff_y - dvv - dvv)

    return v_star_grid


def update_pressure(u_grid, v_grid, p_grid, dx, dy, dt, rho):
    return p_grid


#@jit
def update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho):

    u_updated = np.zeros(u_grid.shape)
    for i in range(1, u_grid.shape[0]-1):
        for j in range(1, u_grid.shape[1]-1):
            print(i)
            u_updated = u_grid[i][j] - dt/(rho*dx) * (p_grid[i+1][j] - p_grid[i][j])
            pass

    v_updated = np.zeros(v_grid.shape)
    for i in range(1, v_grid.shape[0]-1):
        for j in range(1, v_grid.shape[1]-1):
            #v_updated = v_grid[i][j] - dt/(rho*dx) * (p_grid[i][j+1] - p_grid[i][j])
            pass
    
    return u_updated, v_updated


if __name__ == '__main__':

    # Simulation parameters
    dx = 0.001
    dy = 0.001
    dt = 0.001
    time = 30
    timesteps = int(time/dt)

    #TODO: Check CFL

    # System parameters
    Lx = 0.055
    Ly = 0.03
    #TODO: This need to be changed
    rho = 1.
    nu = 1.

    u_grid = np.zeros((int(Lx/dx)+1, int(Ly/dy)))
    v_grid = np.zeros((int(Lx/dx), int(Ly/dy)+1))
    p_grid = np.zeros((int(Lx/dx), int(Ly/dy)))

    for t in tqdm(range(timesteps)):
        u_grid = x_momentum(u_grid, v_grid, dx, dy, dt, nu)
        v_grid = y_momentum(u_grid, v_grid, dx, dy, dt, nu)

        p_grid = update_pressure(u_grid, v_grid, p_grid, dx, dy, dt, rho)

        u_grid, v_grid = update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho)
    pass
