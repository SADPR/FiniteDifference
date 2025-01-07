import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import DT, NUM_STEPS, NUM_CELLS, GRID_X, GRID_Y, W0
from hypernet2D import load_or_compute_snaps, plot_snaps, inviscid_burgers_pod_rbf_2D_global_no_norm


def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, snaps in enumerate(snaps_to_plot):
        plot_snaps(GRID_X, GRID_Y, snaps, inds_to_plot,
                   label=labels[i],
                   fig_ax=(fig, ax1, ax2),
                   color=colors[i],
                   linewidth=linewidths[i])


def main(mu1=4.875, mu2=0.015, save_npy=False, save_plot=False):
    # Use the grid and initial conditions directly from the config
    grid_x, grid_y = GRID_X, GRID_Y
    w0 = W0

    # Load the High-Dimensional Model (HDM) snapshots for the target parameter combination
    snap_folder = 'param_snaps'
    mu = [mu1, mu2]
    hdm_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)

    # Load the global training data for RBF interpolation
    model_dir = "pod_rbf_global_model"
    try:
        with open(os.path.join(model_dir, 'global_weights.pkl'), 'rb') as f:
            data = pickle.load(f)
            W_global = data['W']  # Global weight matrix
            q_p_train = data['q_p_train']
            q_s_train = data['q_s_train']
            epsilon = data['epsilon']
            kernel_name = data['kernel_name']
        print("Loaded global training data successfully.")
    except FileNotFoundError:
        print(f"Training data file '{model_dir}/global_weights.pkl' not found.")
        exit(1)

    # Load the POD basis matrices
    try:
        U_p = np.load(os.path.join(model_dir, 'U_p.npy'))
        U_s = np.load(os.path.join(model_dir, 'U_s.npy'))
        print("Loaded POD basis matrices (U_p and U_s) successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print(f"mu1: {mu1}, epsilon: {epsilon}, kernel: {kernel_name}")

    # Time-stepping to compute the Reduced-Order Model (ROM)
    t0 = time.time()
    pod_rbf_prom_snaps, rbf_times = inviscid_burgers_pod_rbf_2D_global_no_norm(
        grid_x, grid_y, w0, DT, NUM_STEPS, mu, U_p, U_s, W_global, q_p_train, q_s_train, epsilon, kernel_type=kernel_name
    )
    elapsed_time = time.time() - t0
    rbf_its, rbf_jac, rbf_res, rbf_ls = rbf_times

    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')
    print(f'rbf_its: {rbf_its:.2f}, rbf_jac: {rbf_jac:.2f}, rbf_res: {rbf_res:.2f}, rbf_ls: {rbf_ls:.2f}')

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - pod_rbf_prom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Optional: Save the snapshot to a file
    if save_npy:
        snapshot_filename = f'pod_rbf_prom_global_snaps_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.npy'
        np.save(snapshot_filename, pod_rbf_prom_snaps)
        print(f'Snapshot saved as {snapshot_filename}')

    # Optional: Plot and save the results
    if save_plot:
        inds_to_plot = range(0, NUM_STEPS + 1, 100)
        snaps_to_plot = [hdm_snaps, pod_rbf_prom_snaps]
        labels = ['HDM', 'POD-RBF PROM (Global)']
        colors = ['black', 'green']
        linewidths = [2, 1]
        compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)

        plt.tight_layout()
        plt.grid()
        plt.legend(loc=2)
        plot_filename = f'plot_pod_rbf_prom_global_mu1_{mu[0]:.2f}_mu2_{mu[1]:.3f}.png'
        plt.savefig(plot_filename, dpi=300)
        print(f'Plot saved as {plot_filename}')
        plt.show()

    return elapsed_time, relative_error


if __name__ == "__main__":
    main(save_npy=False, save_plot=True)
