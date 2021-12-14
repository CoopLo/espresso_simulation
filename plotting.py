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


def concentration_plots(output_dir, tstep, bcs, dt, save_dir=None):
    print(output_dir)
    fig, ax = plt.subplots(nrows=len(tstep))
    for i in range(len(tstep)):
        concentration = np.loadtxt("./{}/data/concentration_{}.csv".format(output_dir, tstep[i]),
                               delimiter=',')
        concentration[0] = 0
        concentration[-1] = 0
        concentration[:,0] = 0
        concentration[:,-1] = 0
        im = ax[i].imshow(concentration[:,::-1].T, vmin=0, vmax=1, cmap="YlOrBr")

        #ax[i].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
        #ax[i].set_xticklabels(['' for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_xticks([])#i for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_xticklabels([])# for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_title("Time = {}s".format(tstep[i]*dt), y=0.93, fontsize=9)

    ax[-1].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
    ax[-1].set_xticklabels([str(i) for i in [0, 10, 20, 30]])
    ax[-1].set_xlabel("X (mm)")
    fig.text(0.32, 0.45, "Y (mm)", rotation='vertical')
    cbar_ax = fig.add_axes([0.67, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Concentration Over Time", fontsize=12, y=0.98)
    plt.savefig("./concentration_over_time.png", bbox_inches="tight")
    #plt.show()


def pressure_plots(output_dir, tstep, bcs, dt, factor=10, save_dir=None):
    print(output_dir)
    fig, ax = plt.subplots(nrows=len(tstep))
    for i in range(len(tstep)):
        p_grid = np.loadtxt("./{}/data/p_grid_{}.csv".format(output_dir, tstep[i]),
                            delimiter=',') * factor
        fp = np.loadtxt("./{}/data/p_grid_150000.csv".format(output_dir), delimiter=',') * factor
        pressure_min = min([np.min(p_grid), np.min(fp)])
        pressure_max = min([np.max(p_grid), np.max(fp)])

        im = ax[i].imshow(p_grid[:,::-1].T, vmin=pressure_min, vmax=pressure_max, cmap="Reds")

        #ax[i].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
        #ax[i].set_xticklabels(['' for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_xticks([])#i for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_xticklabels([])# for i in [-0.5, 9.5, 19.5, 29.5]])
        ax[i].set_title("Time = {}s".format(tstep[i]*dt), y=0.93, fontsize=9)

    ax[-1].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
    ax[-1].set_xticklabels([str(i) for i in [0, 10, 20, 30]])
    ax[-1].set_xlabel("X (mm)")
    fig.text(0.32, 0.45, "Y (mm)", rotation='vertical')
    cbar_ax = fig.add_axes([0.67, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Pressure Over Time (bar)", fontsize=12, y=0.98)
    plt.savefig("./pressure_over_time.png", bbox_inches="tight")
    #plt.show()


    
def velocity_plots(output_dir, tstep, bcs, dt, factor=10, save_dir=None):
    print(output_dir)
    fig_u, ax_u = plt.subplots(nrows=len(tstep))
    fig_v, ax_v = plt.subplots(nrows=len(tstep))
    for i in range(len(tstep)):

        u_grid = np.loadtxt("./{}/data/u_grid_{}.csv".format(output_dir, tstep[i]),
                            delimiter=',') / factor
        v_grid = np.loadtxt("./{}/data/v_grid_{}.csv".format(output_dir, tstep[i]),
                            delimiter=',') / factor
    
        # GET MIN AND MAX FOR EACH COLORBAR
        fu = np.loadtxt("./{}/data/u_grid_150000.csv".format(output_dir), delimiter=',') / factor
        fv = np.loadtxt("./{}/data/v_grid_150000.csv".format(output_dir), delimiter=',') / factor

        u_final, v_final = final_velocities(u_grid, v_grid, bcs)
        fu_final, fv_final = final_velocities(fu, fv, bcs)

        #print(np.min(u_final), np.min(fu_final), np.min(u_grid), np.min(fu))
        #raise
        v_velocity_min = min([np.min(v_final), np.min(fv_final)])
        v_velocity_max = max([np.max(v_final), np.max(fv_final)])

        u_velocity_min = min([np.min(u_final), np.min(fu_final)])
        u_velocity_max = max([np.max(u_final), np.max(fu_final)])

        u_im = ax_u[i].imshow(u_final[:,::-1].T, vmin=u_velocity_min, vmax=u_velocity_max,
                            cmap="bwr")
        ax_u[i].set_xticks([])#i for i in [-0.5, 9.5, 19.5, 29.5]])
        ax_u[i].set_xticklabels([])# for i in [-0.5, 9.5, 19.5, 29.5]])
        ax_u[i].set_title("Time = {}s".format(tstep[i]*dt), y=0.93, fontsize=9)

        v_im = ax_v[i].imshow(v_final[:,::-1].T, vmin=v_velocity_min, vmax=v_velocity_max,
                            cmap="Reds")
        ax_v[i].set_xticks([])#i for i in [-0.5, 9.5, 19.5, 29.5]])
        ax_v[i].set_xticklabels([])# for i in [-0.5, 9.5, 19.5, 29.5]])
        ax_v[i].set_title("Time = {}s".format(tstep[i]*dt), y=0.93, fontsize=9)

    ax_u[-1].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
    ax_u[-1].set_xticklabels([str(i) for i in [0, 10, 20, 30]])
    ax_u[-1].set_xlabel("X (mm)")
    fig_u.text(0.32, 0.45, "Y (mm)", rotation='vertical')

    cbar_ax_u = fig_u.add_axes([0.67, 0.15, 0.02, 0.7])
    u_cbar = fig_u.colorbar(u_im, cax=cbar_ax_u)
    ubar_ticks = u_cbar.get_ticks()
    u_cbar.set_ticks([i for i in ubar_ticks])
    u_cbar.set_ticklabels(["{0:.0e}".format(i) for i in ubar_ticks])

    fig_u.suptitle("X-Velocity Over Time (m/s)", fontsize=12, y=0.98)
    fig_u.savefig("./x_velocity_over_time.png", bbox_inches="tight")

    ax_v[-1].set_xticks([i for i in [-0.5, 9.5, 19.5, 29.5]])
    ax_v[-1].set_xticklabels([str(i) for i in [0, 10, 20, 30]])
    ax_v[-1].set_xlabel("X (mm)")
    fig_v.text(0.32, 0.45, "Y (mm)", rotation='vertical')

    cbar_ax_v = fig_v.add_axes([0.67, 0.15, 0.02, 0.7])

    v_cbar = fig_v.colorbar(v_im, cax=cbar_ax_v)
    vbar_ticks = v_cbar.get_ticks()
    v_cbar.set_ticks([i for i in vbar_ticks])
    v_cbar.set_ticklabels(["{0:.0e}".format(i) for i in vbar_ticks])

    fig_v.suptitle("Y-Velocity Over Time (m/s)", fontsize=12, y=0.98)
    fig_v.savefig("./y_velocity_over_time.png", bbox_inches="tight")

    #plt.show()


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

    sim = "finally_tasty_point_brews"
    sim = "tasty_point_brews"
    #sim = "exact_brews"

    #data = load_data(sim, 150000, bcs)

    # Output plots for full 30 second simulations
    concentration_plots(sim, [0, 50000, 100000, 150000], bcs, dt)
    pressure_plots(sim, [0, 50000, 100000, 150000], bcs, dt)
    velocity_plots(sim, [0, 50000, 100000, 150000], bcs, dt)

    #for t in tqdm(np.arange(0, 150001, 1000)):
    #    if(t == 0):
    #        data = load_data(sim, 1, bcs, save_dir="animate")
    #    else:
    #        data = load_data(sim, t, bcs, save_dir="animate")

