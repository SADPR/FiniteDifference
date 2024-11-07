import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from config import DT, NUM_STEPS, NUM_CELLS, GRID_X, GRID_Y, W0

# Now, import the required functions from hypernet2D
from hypernet2D import load_or_compute_snaps, make_2D_grid, plot_snaps, inviscid_burgers_pod_rbf_2D

#from pod_rbf_utils import compute_rbf_jacobian_nearest_neighbours_dynamic, interpolate_with_rbf_nearest_neighbours_dynamic
#from gauss_newton_rbf import gauss_newton_rbf

# Parameters from previous setup
DT = 0.05
NUM_STEPS = 500
NUM_CELLS_X = 750
NUM_CELLS_Y = 750
XL, XU = 0, 100
YL, YU = 0, 100

def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(GRID_X, GRID_Y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

def main(mu1=4.75, mu2=0.02):
    # Define the grid and initial conditions
    grid_x, grid_y = make_2D_grid(XL, XU, YL, YU, NUM_CELLS_X, NUM_CELLS_Y)
    w0 = np.ones((NUM_CELLS_X * NUM_CELLS_Y * 2,))  # Example initial condition

    # Load the High-Dimensional Model (HDM) snapshots for the target parameter combination
    snap_folder = 'param_snaps'
    mu = [mu1, mu2]
    hdm_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)

    # Load the KDTree and training data for RBF interpolation
    with open('modes/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    # Load the POD basis matrices
    U_p = np.load('modes/U_p.npy')
    U_s = np.load('modes/U_s.npy')
    #U_full = np.hstack((U_p, U_s))

    # Set epsilon and neighbors based on the value of mu1
    if mu1 == 4.75:
        epsilon = 0.01
        neighbors = 20
    elif mu1 == 4.56:
        epsilon = 0.00001
        neighbors = 20
    elif mu1 == 5.19:
        epsilon = 0.0001
        neighbors = 40
    else:
        raise ValueError(f"Unsupported mu1 value: {mu1}")

    print(f"mu1: {mu1}, epsilon: {epsilon}, neighbors: {neighbors}")

    # Parameters for RBF interpolation
    #epsilon = 0.01
    #neighbors = 100
    r = 10  # Number of primary modes

    # Time-stepping to compute the Reduced-Order Model (ROM) using POD-RBF
    t0 = time.time()
    pod_rbf_prom_snaps, man_times = inviscid_burgers_pod_rbf_2D(
        grid_x, grid_y, w0, DT, NUM_STEPS, mu, U_p, U_s, kdtree, q_p_train, q_s_train, epsilon, neighbors
    )
    elapsed_time = time.time() - t0
    man_its, man_jac, man_res, man_ls = man_times

    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - pod_rbf_prom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Save the snapshot to a file
    np.save(f'pod_rbf_prom_snaps_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.npy', pod_rbf_prom_snaps)
    print(f'Snapshot saved as pod_rbf_prom_snaps_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.npy')

    # Plot and compare snapshots (currently commented out)
    '''
    inds_to_plot = range(0, NUM_STEPS + 1, 100)
    snaps_to_plot = [hdm_snaps, pod_rbf_prom_snaps]
    labels = ['HDM', 'POD-RBF']
    colors = ['black', 'green']
    linewidths = [2, 1]
    compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)

    # Save plot and results
    plt.tight_layout()
    plt.grid()
    plt.legend(loc=2)
    plt.savefig(f'plot_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.png', dpi=300)
    '''

    # Return elapsed time and relative error
    return elapsed_time, relative_error


if __name__ == "__main__":
    main()
