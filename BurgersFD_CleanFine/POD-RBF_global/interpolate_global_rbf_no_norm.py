# interpolate_global_rbf_no_norm.py

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure a non-interactive backend is used
import matplotlib.pyplot as plt
import pickle
import os
import time
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps, make_2D_grid, plot_snaps

# Define RBF kernel functions
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    """Inverse Multiquadric (IMQ) RBF kernel function."""
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def multiquadric_rbf(r, epsilon):
    """Multiquadric (MQ) RBF kernel function."""
    return np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    """Linear RBF kernel function."""
    return r

# Dictionary mapping kernel names to functions
rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'multiquadric': multiquadric_rbf,
    'linear': linear_rbf
}

def reconstruct_snapshot_with_global_rbf_no_norm(
    snapshot, U_p, U_s, q_p_train, W,
    epsilon, kernel_func, print_times=False
):
    """
    Reconstruct the snapshot using a global RBF interpolation (no normalization).
    This is the same as reconstruct_snapshot_with_global_rbf, but without the scaler step.
    """
    start_total_time = time.time()

    # 1) Project the snapshot onto the POD basis (primary modes)
    q = U_p.T @ snapshot
    q_p = q[:U_p.shape[1], :]

    # 2) Instead of normalizing q_p, we use it directly
    #    For each time step, compute RBF interpolation to get q_s_pred
    reconstructed_snapshots_rbf = []
    num_time_steps = q_p.shape[1]
    for i in range(num_time_steps):
        if print_times:
            print(f"Time step {i+1} of {num_time_steps}")

        # Just use q_p[:, i] directly (no scaler.transform)
        q_p_sample = q_p[:, i]  # shape: (primary_modes,)

        # Compute distances to all training points in q_p_train
        # q_p_train has shape (num_train_snapshots, primary_modes)
        dists = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)

        # Apply the selected RBF kernel function
        rbf_values = kernel_func(dists, epsilon)

        # Interpolation => q_s_pred
        q_s_pred = W.T @ rbf_values  # shape: (secondary_modes,)

        # Reconstruct the snapshot
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).T
    if print_times:
        print(f"Reconstruction process completed in {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.875, 0.015]  # Example: mu1=5.19, mu2=0.026

    # Define simulation parameters
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 750, 750
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    # Initial condition (replace with actual initial condition as needed)
    w0 = np.ones((num_cells_x * num_cells_y * 2,))  # Example initial condition

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Ensure the snapshot folder exists
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        print("Please add the required snapshot files before running the script again.")
        exit(1)  # Exit since no snapshots are available

    # Load the specific snapshot for the target parameter pair
    try:
        hdm_snap = load_or_compute_snaps(target_mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        print(f"Loaded HDM snapshot for mu1={target_mu[0]}, mu2={target_mu[1]}")
    except FileNotFoundError as e:
        print(e)
        exit(1)  # Exit since the target snapshot is not available

    # Load the global weight matrix and training data
    try:
        with open('pod_rbf_global_model/global_weights.pkl', 'rb') as f:
            data = pickle.load(f)
            W = data['W']
            q_p_train = data['q_p_train']  # unscaled primary training coords
            epsilon = data['epsilon']
            kernel_name = data.get('kernel_name', 'gaussian')  # Default to 'gaussian' if not provided
        print("Global weight matrix and data loaded successfully.")
    except FileNotFoundError:
        print("File 'pod_rbf_global_model/global_weights.pkl' not found.")
        exit(1)
    except KeyError as e:
        print(f"Key error: {e}. Check if the key names in 'global_weights.pkl' match.")
        exit(1)

    # Select the kernel function
    if kernel_name in rbf_kernels:
        kernel_func = rbf_kernels[kernel_name]
        print(f"Using kernel: {kernel_name}")
    else:
        print(f"Kernel '{kernel_name}' not recognized. Available kernels: {list(rbf_kernels.keys())}")
        exit(1)

    # Load U_p, U_s, and the full U matrix
    try:
        U_p = np.load('pod_rbf_global_model/U_p.npy')
        U_s = np.load('pod_rbf_global_model/U_s.npy')
        U_full = np.hstack((U_p, U_s))  # Full U matrix with all modes
        print("POD basis matrices (U_p and U_s) loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Additional parameters
    num_modes = U_p.shape[1] + U_s.shape[1]
    compare_pod = True  # Set to True to include Standard POD reconstruction
    print_times = False

    # Reconstruct the snapshot using global RBF interpolation (no normalization)
    pod_rbf_reconstructed = reconstruct_snapshot_with_global_rbf_no_norm(
        hdm_snap, U_p, U_s, q_p_train, W, epsilon, kernel_func, print_times
    )

    # Reconstruct the snapshot using standard POD with all modes
    def reconstruct_snapshot_with_pod(snapshot, U, num_modes, print_times=False):
        """Reconstruct a snapshot using standard POD."""
        start_time = time.time()
        U_modes = U[:, :num_modes]
        q_pod = U_modes.T @ snapshot
        reconstructed_pod = U_modes @ q_pod
        if print_times:
            print(f"POD reconstruction took: {time.time() - start_time:.6f} seconds")
        return reconstructed_pod

    pod_reconstructed = None
    if compare_pod:
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, U_full, num_modes, print_times)

    # Save the reconstructed data
    results_dir = "FOM_vs_POD-RBF_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pod_rbf_file_path = os.path.join(
        results_dir, f"reconstructed_snapshot_pod_rbf_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
    )
    np.save(pod_rbf_file_path, pod_rbf_reconstructed)
    print(f"POD-RBF reconstructed snapshot saved successfully to {pod_rbf_file_path}")

    if compare_pod:
        pod_file_path = os.path.join(
            results_dir, f"reconstructed_snapshot_pod_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
        )
        np.save(pod_file_path, pod_reconstructed)
        print(f"Standard POD reconstructed snapshot saved successfully to {pod_file_path}")

    # Calculate and compare reconstruction errors
    pod_rbf_error = np.linalg.norm(hdm_snap - pod_rbf_reconstructed) / np.linalg.norm(hdm_snap)
    print(f"POD-RBF Reconstruction error: {pod_rbf_error:.6e}")

    if compare_pod and pod_reconstructed is not None:
        pod_error = np.linalg.norm(hdm_snap - pod_reconstructed) / np.linalg.norm(hdm_snap)
        print(f"Standard POD Reconstruction error: {pod_error:.6e}")

    # Define indices to plot (e.g., specific time steps)
    inds_to_plot = range(0, num_steps + 1, 100)  # Example: every 100 time steps

    # Prepare snapshots to plot
    snaps_to_plot = [hdm_snap, pod_rbf_reconstructed]
    labels = ['HDM', 'POD-RBF']
    colors = ['black', 'green']
    linewidths = [2, 2]

    if compare_pod and pod_reconstructed is not None:
        snaps_to_plot.append(pod_reconstructed)
        labels.append('Standard POD')
        colors.append('blue')
        linewidths.append(1)

    # Plot the comparison using subplots for x and y slices
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(
            grid_x, grid_y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax1, ax2),
            color=color,
            linewidth=lw
        )

    # Print relative errors
    relative_error = 100 * pod_rbf_error
    print('Relative error (POD-RBF): {:3.2f}%'.format(relative_error))

    if compare_pod and pod_reconstructed is not None:
        relative_error_pod = 100 * pod_error
        print('Relative error (Standard POD): {:3.2f}%'.format(relative_error_pod))

    # Finalize and save the plot
    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper right')
    plot_filename = f"plot_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}_n{num_modes}.png"
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")

    # Optionally, display the plot
    # plt.show()
