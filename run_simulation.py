import numpy as np
from numba import jit
from generate_bed import load_bed
from tqdm import tqdm
from matplotlib import pyplot as plt


###
#  x-momentum
###
def _diff_ux(u_gird, i, j, dx, u_bcs):

    # 0 velocities on boundaries
    left = 0 if(any(np.sum([i+1,j] == u_bcs, axis=1)==2)) else u_grid[i+1][j]
    middle = 0 if(any(np.sum([i,j] == u_bcs, axis=1)==2)) else u_grid[i][j]
    right = 0 if(any(np.sum([i-1,j] == u_bcs, axis=1)==2)) else u_grid[i-1][j]

    return 4*nu/(dx**2) * (left - middle + right)


def _diff_uy(u_grid, i, j, dy, u_bcs):

    # 0 velocities on boundaries
    left = 0 if(any(np.sum([i,j+1] == u_bcs, axis=1)==2) or
              not((j+1) < (u_grid.shape[1]-1))) else u_grid[i][j+1]
    middle = 0 if(any(np.sum([i,j] == u_bcs, axis=1)==2)) else u_grid[i][j]
    right = 0 if(any(np.sum([i,j-1] == u_bcs, axis=1)==2)) else u_grid[i][j-1]

    if(j==0): # Ghost node on bottom wall
        diff_y = 4*nu/(dy**2) * (left - middle - middle)
    elif(j==u_grid.shape[1]-1): # Ghost node on bottom wall
        diff_y = 4*nu/(dy**2) * (-middle - middle + right)
    else:
        diff_y = 4*nu/(dy**2) * (left - middle + right)

    return diff_y


def _duu(u_grid, i, j, dx, u_bcs):

    # 0 velocities on boundaries
    left = 0 if(any(np.sum([i+1,j] == u_bcs, axis=1)==2)) else u_grid[i+1][j]
    middle = 0 if(any(np.sum([i,j] == u_bcs, axis=1)==2)) else u_grid[i][j]
    right = 0 if(any(np.sum([i-1,j] == u_bcs, axis=1)==2)) else u_grid[i-1][j]

    return 1/dx * (((left+middle)/2)**2 - \
                   ((middle + right)/2)**2)


def _dvu(u_grid, v_grid, i, j, dy, u_bcs, v_bcs):

    # 0 velocities on boundaries
    u_left = 0 if(any(np.sum([i,j+1] == u_bcs, axis=1)==2) or
              not((j+1) < (u_grid.shape[1]-1))) else u_grid[i][j+1]
    u_middle = 0 if(any(np.sum([i,j] == u_bcs, axis=1)==2)) else u_grid[i][j]
    u_right = 0 if(any(np.sum([i,j-1] == u_bcs, axis=1)==2)) else u_grid[i][j-1]

    # 0 velocities on boundaries
    v_top_left = 0 if(any(np.sum([(i-1)+1,j] == v_bcs, axis=1)==2)) else v_grid[(i-1)+1][j]
    v_top_right = 0 if(any(np.sum([(i-1),j] == v_bcs, axis=1)==2)) else v_grid[(i-1)][j]
    v_bottom_left = 0 if(any(np.sum([(i-1),(j-1)] == v_bcs, axis=1)==2)) else v_grid[(i-1)][j-1]
    v_bottom_right = 0 if(any(np.sum([(i-1),(j-1)] == v_bcs, axis=1)==2)) else v_grid[(i-1)][j-1]

    if(j==0): # Ghost node on bottom wall
        dvu = 1/dy * (((v_top_left+v_top_right)/2)*\
                  ((u_left+u_middle)/2) - \
                  ((v_bottom_left+v_bottom_right)/2)*\
                  ((u_middle-u_middle)/2))
    elif(j==u_grid.shape[1]-1): # Ghost node on bottom wall
        dvu = 1/dy * (((v_top_left+v_top_right)/2)*\
                  ((-u_middle+u_middle)/2) - \
                  ((v_bottom_left+v_bottom_right)/2)*\
                  ((u_middle+u_right)/2))
    else:
        dvu = 1/dy * (((v_top_left+v_top_right)/2)*\
                  ((u_right+u_middle)/2) - \
                  ((v_bottom_left+v_bottom_right)/2)*\
                  ((u_middle+u_right)/2))
    #if(j==0): # Ghost node on bottom wall
    #    dvu = 1/dy * (((v_grid[(i-1)+1][j]+v_grid[(i-1)][j])/2)*\
    #              ((u_grid[i][j+1]+u_grid[i][j])/2) - \
    #              ((v_grid[(i-1)+1][j-1]+v_grid[(i-1)][j-1])/2)*\
    #              ((u_grid[i][j]-u_grid[i][j])/2))
    #elif(j==u_grid.shape[1]-1): # Ghost node on bottom wall
    #    dvu = 1/dy * (((v_grid[(i-1)+1][j]+v_grid[(i-1)][j])/2)*\
    #              ((-u_grid[i][j]+u_grid[i][j])/2) - \
    #              ((v_grid[(i-1)+1][j-1]+v_grid[(i-1)][j-1])/2)*\
    #              ((u_grid[i][j]+u_grid[i][j-1])/2))
    #else:
    #    dvu = 1/dy * (((v_grid[(i-1)+1][j]+v_grid[(i-1)][j])/2)*\
    #              ((u_grid[i][j+1]+u_grid[i][j])/2) - \
    #              ((v_grid[(i-1)+1][j-1]+v_grid[(i-1)][j-1])/2)*\
    #              ((u_grid[i][j]+u_grid[i][j-1])/2))

    return dvu


#@jit
def x_momentum(u_grid, v_grid, dx, dy, dt, nu, bcs):
    u_star_grid = np.zeros(u_grid.shape)

    for i in range(1, u_grid.shape[0]-1):
        for j in range(u_grid.shape[1]):

            # Diffusion terms
            diff_x = _diff_ux(u_grid, i, j, dx, u_bcs)
            diff_y = _diff_uy(u_grid, i, j, dy, u_bcs)

            # Advection terms
            duu = _duu(u_grid, i, j, dx, u_bcs)
            dvu = _dvu(u_grid, v_grid, i, j, dy, u_bcs, v_bcs)

            # Update grid
            u_star_grid[i][j] = u_grid[i][j] + dt*(diff_x + diff_y - duu - dvu)

    u_star_grid[0] = 0
    u_star_grid[-1] = 0
    u_star_grid[:,0] = 0
    u_star_grid[:,-1] = 0
    return u_star_grid


###
#  y-momentum
###
def _diff_vx(v_grid, i, j, dx, v_bcs):

    # 0 velocities on boundaries
    # Additional condition skips edge because it isn't used in diff_x
    left = 0 if(any(np.sum([i+1,j] == v_bcs, axis=1)==2) or 
             not((i+1) < (v_grid.shape[0]-1))) else v_grid[i+1][j]
    middle = 0 if(any(np.sum([i,j] == v_bcs, axis=1)==2)) else v_grid[i][j]
    right = 0 if(any(np.sum([i-1,j] == v_bcs, axis=1)==2)) else v_grid[i-1][j]

    if(i == 0): # Ghost node left wall
        diff_x = 4*nu/(dx**2) * (left - middle - middle)
    elif(i == v_grid.shape[0]-1): # Ghost node right wall
        diff_x = 4*nu/(dx**2) * (-middle - middle + right)
    else:
        diff_x = 4*nu/(dx**2) * (left - middle + right)

    return diff_x


def _diff_vy(v_grid, i, j, dy, v_bcs):
    
    # 0 velocities on boundaries
    left = 0 if(any(np.sum([i,j+1] == v_bcs, axis=1)==2)) else v_grid[i][j+1]
    middle = 0 if(any(np.sum([i,j] == v_bcs, axis=1)==2)) else v_grid[i][j]
    right = 0 if(any(np.sum([i,j-1] == v_bcs, axis=1)==2)) else v_grid[i][j-1]

    return 4*nu/(dy**2) * (left - middle + right)


def _dvv(v_grid, i, j, dy, v_bcs):

    # 0 velocities on boundaries
    left = 0 if(any(np.sum([i,j+1] == v_bcs, axis=1)==2)) else v_grid[i][j+1]
    middle = 0 if(any(np.sum([i,j] == v_bcs, axis=1)==2)) else v_grid[i][j]
    right = 0 if(any(np.sum([i,j-1] == v_bcs, axis=1)==2)) else v_grid[i][j-1]

    return  1/(2*dy) * (((left+middle)/2)**2 - \
                        ((middle+right)/2)**2)


def _duv(v_grid, u_grid, i, j, dx, v_bcs, u_bcs):

    # 0 velocities on boundaries
    v_left = 0 if(any(np.sum([i,j+1] == v_bcs, axis=1)==2)) else v_grid[i][j+1]
    v_middle = 0 if(any(np.sum([i,j] == v_bcs, axis=1)==2)) else v_grid[i][j]
    v_right = 0 if(any(np.sum([i,j-1] == v_bcs, axis=1)==2)) else v_grid[i][j-1]

    # 0 velocities on boundaries
    u_top_left = 0 if(any(np.sum([i,(j-1)+1] == v_bcs, axis=1)==2)) else u_grid[i][(j-1)+1]
    u_top_right = 0 if(any(np.sum([i,(j-1)] == v_bcs, axis=1)==2)) else u_grid[i][(j-1)]
    u_bottom_left = 0 if(any(np.sum([i-1,j] == v_bcs, axis=1)==2)) else \
                      u_grid[i-1][j]
    u_bottom_right = 0 if(any(np.sum([i-1,(j-1)-1] == v_bcs, axis=1)==2)) else \
                      u_grid[i-1][j-1]

    if(i == 0): # Ghost node left wall
        duv = 1/dx * (((u_top_left+u_top_right)/2)*\
                  ((v_left+v_middle)/2) - \
                  ((u_bottom_left+u_bottom_right)/2)*\
                  ((v_middle-v_middle)/2))
    elif(i == v_grid.shape[0]-1): # Ghost node right wall
        duv = 1/dx * (((u_top_left+u_top_right)/2)*\
                  ((-v_middle+v_middle)/2) - \
                  ((u_bottom_left+u_bottom_right)/2)*\
                  ((v_grid[i][j]+v_grid[i-1][j])/2))
    else:
        duv = 1/dx * (((u_top_left+u_top_right)/2)*\
                  ((v_left+v_right)/2) - \
                  ((u_bottom_left+u_bottom_right)/2)*\
                  ((v_middle+v_right)/2))

    return duv


#@jit
def y_momentum(u_grid, v_grid, dx, dy, dt, nu, bcs):
    v_star_grid = np.copy(v_grid)

    # Need ghost nodes for BCs on walls
    for i in range(v_grid.shape[0]):
        for j in range(1, v_grid.shape[1]-1):

            # Diffusion terms
            diff_x = _diff_vx(v_grid, i, j, dx, v_bcs)
            diff_y = _diff_vy(v_grid, i, j, dy, v_bcs)

            # Advection terms
            dvv = _dvv(v_grid, i, j, dy, v_bcs)
            duv = _duv(v_grid, u_grid, i, j, dx, v_bcs, u_bcs)

            # Update grid
            v_star_grid[i][j] = v_grid[i][j] + dt*(diff_x + diff_y - dvv - dvv)

    # No slip on walls -> Need ghost nodes
    v_star_grid[0] = 0
    v_star_grid[-1] = 0
    v_star_grid[:,0] = v_star_grid[:,1]
    return v_star_grid


#@jit
def _build_matrices(p_grid, x_shape, y_shape, w):
    D_inv = -1/(2*(1/(dy**2) + 1/(dx**2)))*np.eye(p_grid.shape[0])

    # Build block-diagonal T
    A = (-np.eye(y_shape, k=1) - np.eye(y_shape, k=-1)) / (dy**2)
    B = (-np.eye(y_shape)) / (dx**2)
    T = np.zeros((p_grid.shape[0], p_grid.shape[0]))

    T[:y_shape,:y_shape] = A
    T[-y_shape:,-y_shape:] = A
    for i in range(x_shape-1):
        T[(i+1)*y_shape:(i+2)*y_shape, i*y_shape:(i+1)*y_shape] = B
        T[i*y_shape:(i+1)*y_shape, (i+1)*y_shape:(i+2)*y_shape] = B
        T[i*y_shape:(i+1)*y_shape, i*y_shape:(i+1)*y_shape] = A

    U = np.triu(T)
    L = np.tril(T)

    T = np.copy(U)
    left_matrix = np.linalg.inv(np.eye(p_grid.shape[0]) - w*np.dot(D_inv, L))
    D_inv_U = np.dot(D_inv, U)

    return left_matrix, D_inv, D_inv_U


def _multiply(left_matrix, p_grid, D_inv_U, D_inv, b, w):
    return np.dot(left_matrix,np.dot((1-w)*np.eye(p_grid.shape[0]) + w*D_inv_U, p_grid)) + \
           np.dot(left_matrix, w*np.dot(D_inv, b))


def update_pressure(u_grid, v_grid, p_grid, dx, dy, dt, rho, x_shape, y_shape, bcs,
                    w=1.8, threshold=1e-5, OVERPRESSURE=9.):

    final_grid = np.zeros(p_grid.shape)

    # Get matrices
    left_matrix, D_inv, D_inv_U = _build_matrices(p_grid, x_shape, y_shape, w)

    # Run iterations until convergence
    old_grid = np.ones(p_grid.shape) + 100
    right_bc =  [i*dy/(dx**2) for i in range(x_shape)]
    it = 0

    # Set up b matrix for poisson equation
    b_matrix = np.zeros((u_grid.shape[0]-1, u_grid.shape[1]))
    for i in range(1,b_matrix.shape[0]-1):
        for j in range(1,b_matrix.shape[1]-1):
            #TODO Make sure indexing is correct here -> this stays constant during iteration
            b_matrix[i][j] = rho/dt * ((u_grid[i][j] - u_grid[i-1][j])/dx + \
                                       (v_grid[i][j] - v_grid[i][j-1])/dy)
    velocities = b_matrix[1:-1,1:-1].reshape(p_grid.shape)
    #print(velocities)
    assert velocities.shape == p_grid.shape

    it = 1

    # Get index grid for coffee BCs
    idx_grid = np.zeros((u_grid.shape[0]-1, u_grid.shape[1]), dtype=int)
    for i in range(idx_grid.shape[0]):
        for j in range(idx_grid.shape[1]):
            idx_grid[i][j] = idx_grid.shape[1]*i + j
    idxs = idx_grid[1:-1,1:-1].reshape(-1,1)

    # Convert to indices of reshaped subgrid
    coffee_bcs = []
    for b in bcs:
        coffee_bcs.append(np.argwhere(idxs == idx_grid[b[0], b[1]])[0][0])

    while(np.linalg.norm(old_grid - p_grid) > threshold):
        it += 1
        #print(np.linalg.norm(old_grid - p_grid))
        if(it > 1000):
            print(np.linalg.norm(old_grid - p_grid))
            raise

        # Update boundary conditions
        b = np.copy(velocities)
        b[-y_shape:] -= p_grid[-y_shape:]/(dx**2) # Right bc
        b[:y_shape] -= p_grid[:y_shape]/(dx**2) # Left bc
        b[0::y_shape] -= 0 # Bottom has zero pressure, not necessary
        b[y_shape-1::y_shape] -= OVERPRESSURE/(dy**2) # top

        #b[-y_shape:] -= 0 # Test x-velocity
        #b[:y_shape] -= OVERPRESSURE/(dy**2) # Test x-velocity
        #b[0::y_shape] -= p_grid[0::y_shape]/(dy**2) # Test x-velocity
        #b[y_shape-1::y_shape] -= p_grid[y_shape-1::y_shape]/(dy**2) # Test x-velocity

        # Copy grid and do multiplications
        old_grid = np.copy(p_grid)
        p_grid = _multiply(left_matrix, p_grid, D_inv_U, D_inv, b, w)

    print("ITERATIONS: {}".format(it))
    return p_grid


def point_jacobi(u_grid, v_grid, p_grid, dx, dy, dt, rho, x_shape, y_shape, bcs,
                 threshold=1e-12, OVERPRESSURE=9.):
    final_grid = np.copy(p_grid)
    grid = p_grid[1:-1,1:-1].reshape(-1,)

    S_inv = -1/(2*(1/(dy**2) + 1/(dx**2))) # Diagonal matrix can be simplified

    # Build block-diagonal T
    A = (-np.eye(y_shape, k=1) - np.eye(y_shape, k=-1)) / (dx**2)
    B = (-np.eye(y_shape)) / (dy**2)
    T = np.zeros((grid.shape[0], grid.shape[0]))
    T[:y_shape,:y_shape] = A
    T[-y_shape:,-y_shape:] = A
    for i in range(x_shape-1):
        T[(i+1)*y_shape:(i+2)*y_shape, i*y_shape:(i+1)*y_shape] = B
        T[i*y_shape:(i+1)*y_shape, (i+1)*y_shape:(i+2)*y_shape] = B
        T[i*y_shape:(i+1)*y_shape, i*y_shape:(i+1)*y_shape] = A

    # Run iterations until convergence
    old_grid = np.copy(grid) + 100
    num_it = 0
    while(np.linalg.norm(old_grid - grid) > threshold):

        # Update boundary conditions
        b = np.zeros(grid.shape)
        b[-y_shape:] -= grid[-y_shape:]/(dx**2) # Right bc
        b[:y_shape] -= grid[:y_shape]/(dx**2) # Left bc
        #b[-y_shape:] -= OVERPRESSURE/(dx**2 * 3)
        #b[:y_shape] -= OVERPRESSURE/(dx**2 * 3)
        b[0::y_shape] -= 0 # Bottom has zero pressure, not necessary
        b[y_shape-1::y_shape] -= OVERPRESSURE/(dy**2) # top
        temp_grid = np.zeros(p_grid.shape)
        #print(temp_grid.shape)
        for i in range(temp_grid.shape[0]):
            for j in range(temp_grid.shape[1]):
                temp_grid[i][j] = temp_grid.shape[1]*i + j
        temp_grid = temp_grid[1:-1,1:-1].reshape(-1,1)

        # Copy grid and do multiplications
        old_grid = np.copy(grid)
        grid = S_inv * (np.dot(T, grid) + b)

        num_it += 1

    print("NUMBER OF ITERATIONS: {}".format(num_it))

    final_grid[1:-1,1:-1] = grid.reshape((x_shape,y_shape))
    return final_grid


#@jit
def update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho, u_bcs, v_bcs):

    # Bulk
    u_updated = np.copy(u_grid)
    for i in range(1, u_grid.shape[0]-1):
        for j in range(1, u_grid.shape[1]-1):
            u_updated[i][j] = u_grid[i][j] - dt/(rho) * \
                              (p_grid[(i-1)+1][j] - p_grid[(i-1)][j])

    # Bulk
    v_updated = np.copy(v_grid)
    for i in range(1, v_grid.shape[0]-1):
        for j in range(1, v_grid.shape[1]-1):
            v_updated[i][j] = v_grid[i][j] + dt/(rho) * \
                              (p_grid[i][(j-1)+1] - p_grid[i][(j-1)])

    # Bottom and top velocities are the same as interior
    v_updated[:,-1] = v_updated[:,-2]
    v_updated[:,0] = v_updated[:,1]
    #u_updated[-1] = u_updated[-2]
    #u_updated[0] = u_updated[1]

    for v in v_bcs:
        v_updated[v[0], v[1]] = 0

    for u in u_bcs:
        u_updated[u[0], u[1]] = 0

    return u_updated, v_updated


#@jit
def final_velocities(u_grid, v_grid, bcs):
    u_final = np.zeros((u_grid.shape[0]-1, u_grid.shape[1]))
    v_final = np.zeros((v_grid.shape[0], v_grid.shape[1]-1))

    for i in range(u_grid.shape[0]-1):
        for j in range(u_grid.shape[1]):
            u_final[i][j] = (u_grid[i+1][j] + u_grid[i][j])/2

    for i in range(v_grid.shape[0]):
        for j in range(v_grid.shape[1]-1):
            v_final[i][j] = (v_grid[i][j+1] + v_grid[i][j])/2

    for b in bcs:
        u_final[b[0], b[1]] = 0
        v_final[b[0], b[1]] = 0

    return u_final, v_final


def pretty_plot(u_grid, v_grid, p_grid, grounds, bcs, t):
    u_final, v_final = final_velocities(u_grid, v_grid, bcs=bcs)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
    ax[0][0].imshow(grounds[:,::-1].T)
    ax[0][0].set_title("Espresso Grounds", fontsize=16)
    ax[0][1].imshow(p_grid[:,::-1].T, cmap="Reds")
    ax[0][1].set_title("Pressure", fontsize=16)

    # Set colorbar
    cbar_max = np.max((u_final, v_final))
    cbar_min = np.min((u_final, v_final))

    #ax[1].imshow(u_grid[:,::-1].T)
    ax[1][0].imshow(u_final[:,::-1].T, cmap="Reds", vmin=cbar_min, vmax=cbar_max)
    ax[1][0].set_title("X-Velocity", fontsize=16)
    #ax[2].imshow(v_grid[:,::-1].T)
    im = ax[1][1].imshow(v_final[:,::-1].T, cmap="Reds", vmin=cbar_min, vmax=cbar_max)
    ax[1][1].set_title("Y-Velocity", fontsize=16)

    # Colorbar
    plt.colorbar(im, orientation='vertical')

    # Fix plot ticks
    ax[0][0].set_xticks([i for i in range(0, grounds.shape[0], int(grounds.shape[0]/5))])
    ax[0][0].set_xticklabels(['' for i in \
                             range(0, grounds.shape[0], int(grounds.shape[0]/5))])
    ax[0][1].set_xticks([i for i in range(0, grounds.shape[0], int(grounds.shape[0]/5))])
    ax[0][1].set_xticklabels(['' for i in \
                             range(0, grounds.shape[0], int(grounds.shape[0]/5))])
    ax[0][0].set_yticks([i for i in range(0, grounds.shape[1], int(grounds.shape[1]/5))])
    ax[0][0].set_yticklabels([str(i) for i in \
                              range(0, grounds.shape[1], int(grounds.shape[1]/5))])
    ax[1][0].set_yticks([i for i in range(0, grounds.shape[1], int(grounds.shape[1]/5))])
    ax[1][0].set_yticklabels([str(i) for i in \
                    range(0, grounds.shape[1], int(grounds.shape[1]/5))])
    ax[0][1].set_yticks([i for i in range(grounds.shape[1], int(grounds.shape[1]/5))])
    ax[1][1].set_yticks([i for i in range(grounds.shape[1], int(grounds.shape[1]/5))])

    # Axis labels
    ax[1][0].set_xlabel("X", fontsize=14)
    ax[1][1].set_xlabel("X", fontsize=14)
    ax[0][0].set_ylabel("Y", fontsize=14, rotation=0, x=-0.1)
    ax[1][0].set_ylabel("Y", fontsize=14, rotation=0, x=-0.1)
    fig.suptitle("Timestep: {}".format(t), fontsize=18)
    plt.tight_layout()
    #plt.show()
    zeros = "0"*(int(np.log(100)/np.log(10)) - int(np.log(t+1)/np.log(10)))
    plt.savefig("./brews/{}{}.png".format(zeros, t+1))
    plt.close()
    #raise
    


if __name__ == '__main__':

    # Simulation parameters
    dx = 0.001
    dy = 0.001
    dt = 0.0000001
    time = 30
    timesteps = int(time/dt)

    #TODO: Check CFL

    # System parameters
    #Lx = 0.012
    #Ly = 0.009
    Lx = 0.055
    Ly = 0.03
    #TODO: This need to be changed
    rho = 1.
    rho = 0.2818 # Viscosity of heated water
    nu = 1.
    OVERPRESSURE = 0.001

    u_grid = np.zeros((int(Lx/dx)+1, int(Ly/dy)))
    v_grid = np.zeros((int(Lx/dx), int(Ly/dy)+1))
    p_grid = np.zeros((int(Lx/dx), int(Ly/dy)))
    p_grid[:,-1] = OVERPRESSURE
    #p_grid[0] = OVERPRESSURE # Test x-velocity

    # Load coffee grounds
    #grounds = load_bed(p_grid, 0.09, seed=1, boulder_frac=1.)
    grounds = load_bed(np.copy(p_grid), 0.2, seed=1, boulder_frac=0.3)
    grounds[:,-1] = 0
    bcs = np.argwhere(grounds == 1)

    # Make u_bcs for staggered grid
    temp_u_bcs = np.copy(bcs)
    temp_u_bcs[:,0] += 1
    #temp_u_bcs[:,1] -= 1
    u_bcs = np.concatenate((bcs, temp_u_bcs), axis=0)
    u_bcs -= 1

    # Make v_bcs for staggered grid
    temp_v_bcs = np.copy(bcs)
    temp_v_bcs[:,1] -= 1
    v_bcs = np.concatenate((bcs, temp_v_bcs), axis=0)

    #timesteps = 100000
    timesteps = 50
    for t in tqdm(range(timesteps)):
        u_grid = x_momentum(u_grid, v_grid, dx, dy, dt, nu, bcs=u_bcs)
        v_grid = y_momentum(u_grid, v_grid, dx, dy, dt, nu, bcs=v_bcs)

        # Pressure laplacian is good. Need to figure out velocities before going Poisson
        flat_p_grid = update_pressure(u_grid, v_grid, p_grid[1:-1,1:-1].reshape(-1,1),
                                            dx, dy, dt, rho,
                                            p_grid.shape[0]-2, p_grid.shape[1]-2,
                                            bcs=bcs, OVERPRESSURE=OVERPRESSURE)
        p_grid[1:-1,1:-1] = flat_p_grid.reshape((p_grid.shape[0]-2, p_grid.shape[1]-2))

        # Test pressure BCs with PJ method. I believe its correct now.
        #p_grid = point_jacobi(u_grid, v_grid, p_grid, dx, dy, dt, rho,
        #                      p_grid.shape[0]-2, p_grid.shape[1]-2, OVERPRESSURE=OVERPRESSURE,
        #                      threshold=1e-5)

        # Pressure is same at walls
        p_grid[0] = p_grid[1]
        p_grid[-1] = p_grid[-2]
        #p_grid[:,0] = p_grid[:,1] # Test x-velocity
        #p_grid[:,-1] = p_grid[:,-2] # Test x-velocity
        u_grid, v_grid = update_velocities(u_grid, v_grid, p_grid, dx, dy, dt, rho, u_bcs, v_bcs)

        pretty_plot(u_grid, v_grid, p_grid, grounds, bcs, t)


    # This is my awesome solution to no-flux pressure boundary conditions on the grounds :(
    #temp_grid = np.zeros(p_grid.shape)
    #print(temp_grid.shape)
    #for i in range(temp_grid.shape[0]):
    #    for j in range(temp_grid.shape[1]):
    #        temp_grid[i][j] = temp_grid.shape[1]*i + j
    #print(temp_grid)
    #print(temp_grid.reshape(-1,1).rshape((7,5)))
