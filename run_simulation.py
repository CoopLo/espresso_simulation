import numpy as np
from numba import jit
from generate_bed import load_bed
from tqdm import tqdm


@jit
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

            # dvu/dy TODO: fix this
            dvu = 1/dy * (((v_grid[(i-1)+1][j]+v_grid[(i-1)][j])/2)*\
                          ((u_grid[i][j+1]+u_grid[i][j])/2) - \
                          ((v_grid[(i-1)+1][j-1]+v_grid[(i-1)][j-1])/2)*\
                          ((u_grid[i][j]+u_grid[i][j-1])/2))

            # Update grid
            u_star_grid[i][j] = u_grid[i][j] + dt*(diff_x + diff_y - duu - dvu)

    return u_star_grid


@jit
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
            duv = 1/dy * (((u_grid[i][(j-1)+1]+u_grid[i][(j-1)])/2)*\
                          ((v_grid[i+1][j]+v_grid[i][j])/2) - \
                          ((u_grid[(i-1)-1][j]+u_grid[(i-1)-1][j-1])/2)*\
                          ((v_grid[i][j]+v_grid[i-1][j])/2))

            # Update grid
            v_star_grid[i][j] = v_grid[i][j] + dt*(diff_x + diff_y - dvv - dvv)

    return v_star_grid


@jit
def _build_matrices(p_grid, x_shape, y_shape, w):
    D_inv = -1/(2*(1/(dy**2) + 1/(dx**2)))*np.eye(p_grid.shape[0])

    # Build block-diagonal T
    A = (-np.eye(x_shape, k=1) - np.eye(x_shape, k=-1)) / (dy**2)
    B = (-np.eye(x_shape)) / (dx**2)
    T = np.zeros((p_grid.shape[0], p_grid.shape[0]))

    T[:x_shape,:x_shape] = A
    T[-x_shape:,-x_shape:] = A
    for i in range(y_shape-1):
        #print(i)
        T[(i+1)*x_shape:(i+2)*x_shape, i*x_shape:(i+1)*x_shape] = B
        T[i*x_shape:(i+1)*x_shape, (i+1)*x_shape:(i+2)*x_shape] = B
        T[i*x_shape:(i+1)*x_shape, i*x_shape:(i+1)*x_shape] = A

    U = np.triu(T)
    L = np.tril(T)

    T = np.copy(U)
    left_matrix = np.linalg.inv(np.eye(p_grid.shape[0]) - w*np.dot(D_inv, L))
    D_inv_U = np.dot(D_inv, U)

    return left_matrix, D_inv, D_inv_U


@jit
def _multiply(left_matrix, p_grid, D_inv_U, D_inv, b, w):
    return np.dot(left_matrix,(np.dot((1-w)*np.eye(p_grid.shape[0]) + w*D_inv_U, p_grid) + \
                  w*np.dot(D_inv, b)))


@jit
def update_pressure(u_grid, v_grid, p_grid, dx, dy, dt, rho, x_shape, y_shape,
                    w=1.8, threshold=1e-5):
    final_grid = np.zeros(p_grid.shape)

    left_matrix, D_inv, D_inv_U = _build_matrices(p_grid, x_shape, y_shape, w)

    # Run iterations until convergence
    old_grid = np.ones(p_grid.shape) + 100
    right_bc =  [i*dy/(dx**2) for i in range(x_shape)]
    it = 0
    while(np.linalg.norm(old_grid - p_grid) > threshold):

        # Update boundary conditions
        b = np.zeros(p_grid.shape)
        b[-x_shape:] = 0 # Right bc
        b[:x_shape] = 0 # Left bc
        b[0::y_shape] -= p_grid[y_shape-1::y_shape]/(dy**2) # bottom
        b[y_shape-1::y_shape] -= p_grid[0::y_shape]/(dy**2) # top

        # Copy grid and do multiplications
        old_grid = np.copy(p_grid)
        p_grid = _multiply(left_matrix, p_grid, D_inv_U, D_inv, b, w)

    return p_grid


@jit
def update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho):

    u_updated = np.zeros(u_grid.shape)
    for i in range(1, u_grid.shape[0]-1):
        for j in range(1, u_grid.shape[1]-1):
            u_updated[i][j] = u_grid[i][j] - dt/(rho*dx) * \
                              (p_grid[(i-1)+1][j] - p_grid[(i-1)][j])

    v_updated = np.zeros(v_grid.shape)
    for i in range(1, v_grid.shape[0]-1):
        for j in range(1, v_grid.shape[1]-1):
            v_updated[i][j] = v_grid[i][j] - dt/(rho*dx) * \
                              (p_grid[i][(j-1)+1] - p_grid[i][(j-1)])
    
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
    print(p_grid.shape[0])

    for t in tqdm(range(timesteps)):
        u_grid = x_momentum(u_grid, v_grid, dx, dy, dt, nu)
        v_grid = y_momentum(u_grid, v_grid, dx, dy, dt, nu)

        flat_p_grid = update_pressure(u_grid, v_grid, p_grid[1:-1,1:-1].reshape(-1,1),
                                            dx, dy, dt, rho,
                                            p_grid.shape[0]-2, p_grid.shape[1]-2)
        p_grid[1:-1,1:-1] = flat_p_grid.reshape((p_grid.shape[0]-2, p_grid.shape[1]-2))

        u_grid, v_grid = update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho)
