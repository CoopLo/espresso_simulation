import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

def load_bed(grid, infill, point=1, boulder=3, seed=None):

    if(seed is not None):
        np.random.seed(seed)

    allowed_infill = int(np.prod(grid.shape)*infill)

    # Get number of boulders
    num_boulder = int(allowed_infill*0.3/boulder**2)

    # Get number of points
    num_point = int(allowed_infill*0.7/point**2)

    # Place boulders, make sure they don't go over the edges, no overlaps
    num_placed = 0
    while(num_placed < num_boulder):
        idx = np.random.randint(1, grid.shape[0]-1)
        jdx = np.random.randint(1, grid.shape[1]-1)
        if((grid[idx:idx+boulder, jdx:jdx+boulder] == 1).any() or \
           (idx >= (grid.shape[0]-2)) or (jdx >= (grid.shape[1]-2))):
            continue
        else:
            grid[idx:idx+boulder, jdx:jdx+boulder] = 1
            num_placed += 1

    # Place points, no overlaps
    num_placed = 0
    while(num_placed < num_point):
        idx = np.random.randint(1, grid.shape[0]-1)
        jdx = np.random.randint(1, grid.shape[1]-1)
        if(grid[idx,jdx] == 1):
            continue
        else:
            grid[idx,jdx] = 1
            num_placed += 1

    return grid

def update_grid(grid, updated_grid, dx, dy, dt, u, v, mu, bc_grid, tstep=0):
    for i in range(1, grid.shape[0]-1):
        for j in range(1, grid.shape[1]-1):

            # If we're at a boundary condition, keep 0 and skip loop
            if(any(np.sum([i,j] == bc_grid, axis=1)==2)):
                updated_grid[i][j] = 0
                continue
            else: 
                pt = (1 - u*dt/dx - v*dt/dy - 2*mu*dt/(dx**2) - 2*mu*dt/(dy**2))*grid[i][j]

            # Contribution from boundary conditions is 0
            right = 0 if(any(np.sum([i,j+1] == bc_grid, axis=1)==2)) else \
                    mu*dt/(dx**2)*grid[i][j+1]
            left = 0 if(any(np.sum([i,j-1] == bc_grid, axis=1)==2)) else \
                   (mu*dt/(dx**2) + u*dt/dx)*grid[i][j-1]
            bottom = 0 if(any(np.sum([i-1,j] == bc_grid, axis=1)==2)) else \
                     (mu*dt/(dy**2))*grid[i-1][j]
            top = 0 if(any(np.sum([i+1,j] == bc_grid, axis=1)==2)) else \
                  (mu*dt/(dy**2) + v*dt/dy)*grid[i+1][j]

            updated_grid[i][j] = pt + top + bottom + left + right

    # Bottom row
    for i in range(1, grid.shape[1]-1):
        updated_grid[1][i] += grid[1][i] * (mu*dt/(dy**2) + v*dt/dy)

    # Top row
    for i in range(1, grid.shape[1]-1):
        updated_grid[-2][i] += grid[-2][i] * (mu*dt/(dx**2))

    # Right Column
    for i in range(1, grid.shape[0]-1):
        updated_grid[i][-2] += grid[i][-2] * mu*dt/(dx**2)

    return updated_grid


if __name__ == '__main__':
    dx = 0.01
    dy = 0.01
    dt = 0.0003

    Lx = 0.5
    Ly = 0.25

    mu = 0.01
    u = 0.0
    v = -1.

    infill = 0.1

    point_size = 0.01/dx
    boulder_size = 0.03/dx

    bc_grid = np.zeros((int(Ly/dy), int(Lx/dx)))

    bc_grid = load_bed(bc_grid, infill, seed=1)
    bc_grid = np.argwhere(bc_grid == 1)

    grid = np.zeros((int(Ly/dy), int(Lx/dx)), dtype=float)
    grid[0] = 1

    timesteps = 10
    for t in tqdm(range(timesteps)):
        grid = update_grid(grid, np.copy(grid), dx, dy, dt, u, v, mu, bc_grid, t)

    # Put grounds into plot
    for v in bc_grid:
        grid[v[0]][v[1]] = -1

    fig, ax = plt.subplots()
    ax.imshow(grid)
    plt.show()

