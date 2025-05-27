# interpolate_global_rbf.py

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
    """Inverse Multiquadric RBF kernel function."""
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def multiquadric_rbf(r, epsilon):
    """Multiquadric RBF kernel function."""
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

def reconstruct_snapshot_with_global_rbf(snapshot, U_p, U_s, q_p_train, W, scaler, epsilon, kernel_func, print_times=False):
    start_total_time = time.time()
    q = U_p.T @ snapshot
    q_p = q[:U_p.shape[1], :]

    # Normalize q_p using the saved scaler
    q_p_normalized = scaler.transform(q_p.T).T  # Note the transpose operations

    reconstructed_snapshots_rbf = []
    num_time_steps = q_p.shape[1]
    for i in range(num_time_steps):
        if print_times:
            print(f"Time step {i+1} of {num_time_steps}")
        q_p_sample = q_p_normalized[:, i]

        # Compute distances to all training points
        dists = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)
        # Apply the selected RBF kernel function
        rbf_values = kernel_func(dists, epsilon)
        # Interpolation
        q_s_pred = W.T @ rbf_values
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).T
    if print_times:
        print(f"Reconstruction process completed in {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.56, 0.019]  # Example: mu1=5.19, mu2=0.026

    # Define simulation parameters
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
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
            q_p_train = data['q_p_train']  # Ensure key matches training script
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

    # Load the saved scaler (Min-Max scaler)
    try:
        with open('pod_rbf_global_model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Min-Max scaler loaded successfully.")
    except FileNotFoundError:
        print("Scaler file 'pod_rbf_global_model/scaler.pkl' not found.")
        exit(1)

    # Additional parameters
    num_modes = U_p.shape[1] + U_s.shape[1]
    compare_pod = True  # Set to True to include Standard POD reconstruction
    print_times = False

    # Reconstruct the snapshot using global RBF interpolation
    pod_rbf_reconstructed = reconstruct_snapshot_with_global_rbf(
        hdm_snap, U_p, U_s, q_p_train, W, scaler, epsilon, kernel_func, print_times
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
    colors = ['black', 'blue']
    linewidths = [3, 3]

    if compare_pod and pod_reconstructed is not None:
        snaps_to_plot.append(pod_reconstructed)
        labels.append('Standard POD')
        colors.append('#9400D3')
        linewidths.append(3)

    # Apply LaTeX-based font settings
    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": ["STIXGeneral"]
    })
    plt.rc('font', size=13)

    # Create figure and subplots (two rows for X and Y slices)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Add a title with parameters and error info
    fig.suptitle(
        f"POD-RBF Projection for $\\mu_1 = {target_mu[0]:.2f}$, $\\mu_2 = {target_mu[1]:.3f}$\n",
        fontsize=14,
        y=0.98
    )

    # Plot the snapshots
    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(
            grid_x, grid_y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax1, ax2),
            color=color,
            linewidth=lw
        )

    # Add legends to subplots
    ax2.legend(loc="center right", fontsize=20, frameon=False)

    # Adjust layout to reduce margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0.2)

    # Save the plot
    plot_filename = f"mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}_pod_rbf_projection.png"
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300, bbox_inches='tight', pad_inches=0.1)

    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")


    # Optionally, display the plot
    # plt.show()
