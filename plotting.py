import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from run_simulation import final_velocities
from generate_bed import load_bed
import os


def load_data(output_dir, tstep, bcs, save_dir=None):
    concentration = np.loadtxt("./{}/data/concentration_{}.csv".format(output_dir, tstep),
                               delimiter=',')
    p_grid = np.loadtxt("./{}/data/p_grid_{}.csv".format(output_dir, tstep), delimiter=',')
    u_grid = np.loadtxt("./{}/data/u_grid_{}.csv".format(output_dir, tstep), delimiter=',')
    v_grid = np.loadtxt("./{}/data/v_grid_{}.csv".format(output_dir, tstep), delimiter=',')

    # GET MIN AND MAX FOR EACH COLORBAR
    fc = np.loadtxt("./{}/data/concentration_150000.csv".format(output_dir),
                    delimiter=',')
    fp = np.loadtxt("./{}/data/p_grid_150000.csv".format(output_dir), delimiter=',')
    fu = np.loadtxt("./{}/data/u_grid_150000.csv".format(output_dir), delimiter=',')
    fv = np.loadtxt("./{}/data/v_grid_150000.csv".format(output_dir), delimiter=',')


    u_final, v_final = final_velocities(u_grid, v_grid, bcs)
    fu_final, fv_final = final_velocities(fu, fv, bcs)

    velocity_min = min([np.min(u_final), np.min(fu_final), np.min(v_final), np.min(fv_final)])
    velocity_max = max([np.max(u_final), np.max(fu_final), np.max(v_final), np.max(fv_final)])

    pressure_min = min([np.min(p_grid), np.min(fp)])
    pressure_max = min([np.max(p_grid), np.max(fp)])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,7))

    # Take out 0 flux BCs
    concentration[0] = 0
    concentration[-1] = 0
    concentration[:,-1] = 0
    ax[0][0].imshow(concentration[:,::-1].T, vmin=0, vmax=1, cmap="YlOrBr")
    ax[0][0].set_title("Espresso Grounds and Concentration", fontsize=16)
    ax[0][1].imshow(p_grid[:,::-1].T, cmap="Reds", vmin=pressure_min, vmax=pressure_max)
    ax[0][1].set_title("Pressure", fontsize=16)

    # Set colorbar
    #cbar_max = np.max((u_final, v_final))
    #cbar_min = np.min((u_final, v_final))

    #ax[1][0].imshow(u_grid[:,::-1].T, cmap="Reds", vmin=cbar_min, vmax=cbar_max)
    ax[1][0].imshow(u_final[:,::-1].T, cmap="Reds", vmin=velocity_min, vmax=velocity_max)
    ax[1][0].set_title("X-Velocity", fontsize=16)

    #im = ax[1][1].imshow(v_grid[:,::-1].T, cmap="Reds", vmin=cbar_min, vmax=cbar_max)
    im = ax[1][1].imshow(v_final[:,::-1].T, cmap="Reds", vmin=velocity_min, vmax=velocity_max)
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
    fig.suptitle("Timestep: {}".format(tstep), fontsize=18)
    plt.tight_layout()

    if(save_dir is None):
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        zeros = "0"*(int(np.log(10000001)/np.log(10)) - int(np.log(tstep)/np.log(10)))
        if(tstep==1000):
            zeros = zeros[:-1]
        plt.savefig("./{}/{}{}.png".format(save_dir, zeros, tstep))
    plt.close()



if __name__ == '__main__':

    # THESE PARAMETERS ARE CONSTANT FOR ALL SIMULATIONS
    FACTOR = 10
    dx = 0.001 * FACTOR # Units are 10 mm
    dy = 0.001 * FACTOR# Units are 10 mm
    dt = 0.00002 * FACTOR
    time = 30
    Lx = 0.03 * FACTOR
    Ly = 0.015 * FACTOR
    timesteps = int(time/dt)

    # Physical constants
    rho = 1.
    nu = 0.2818 / FACTOR
    OVERPRESSURE = 6 / FACTOR


    p_grid = np.zeros((int(Lx/dx), int(Ly/dy)))
    p_grid[:,-1] = OVERPRESSURE
    grounds = load_bed(np.copy(p_grid), 0.3, seed=5, boulder_frac=0.7)
    grounds[:,-1] = 0
    bcs = np.argwhere(grounds == 1)


    # System parameters
    Lx = 0.03 * FACTOR
    Ly = 0.015 * FACTOR

    sim = "tasty_point_brews"

    data = load_data(sim, 150000, bcs)

    #for t in tqdm(np.arange(0, 150001, 1000)):
    #    if(t == 0):
    #        data = load_data(sim, 1, bcs, save_dir="animate")
    #    else:
    #        data = load_data(sim, t, bcs, save_dir="animate")

