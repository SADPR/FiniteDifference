import numpy as np
import sys
import os
import time
from matplotlib import pyplot as plt
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps, make_2D_grid, plot_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

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

def matern_kernel(r, epsilon):
    """Matérn kernel function with nu=3/2.
    
    k(r) = (1 + sqrt(3)*epsilon*r) * exp(-sqrt(3)*epsilon*r)
    """
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * epsilon * r) * np.exp(-sqrt3 * epsilon * r)

# Dictionary mapping kernel names to functions
rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'multiquadric': multiquadric_rbf,
    'linear': linear_rbf,
    'matern': matern_kernel
}

def reconstruct_snapshot_with_global_rbf(snapshot, U_p, U_s, q_p_train, W, scaler, epsilon, kernel_func, print_times=False):
    start_total_time = time.time()
    q = U_p.T @ snapshot
    q_p = q[:U_p.shape[1], :]

    # Normalize q_p using the saved scaler
    q_p_normalized = scaler.transform(q_p.T).T  # Note the transpose operations

    reconstructed_snapshots_rbf = []
    reconstructed_q_s_rbf = []
    num_time_steps = q_p.shape[1]
    for i in range(num_time_steps):
        if print_times:
            print(f"Time step {i+1} of {num_time_steps}")
        q_p_sample = q_p_normalized[:, i]
        # Compute distances to all training points
        dists = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)
        # Apply the selected RBF kernel function
        rbf_values = kernel_func(dists, epsilon)
        # Interpolation to predict secondary coefficients
        q_s_pred = W.T @ rbf_values
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred
        reconstructed_q_s_rbf.append(q_s_pred)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).T
    reconstructed_q_s_rbf = np.array(reconstructed_q_s_rbf).T
    if print_times:
        print(f"Reconstruction process completed in {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf, reconstructed_q_s_rbf

if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.56, 0.019]  # Example: mu1=4.56, mu2=0.019

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

    hdm_snap = load_or_compute_snaps(target_mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
    print(f"Loaded HDM snapshot for mu1={target_mu[0]}, mu2={target_mu[1]}")

    # Load global weights and training data
    with open('pod_rbf_global_model/global_weights.pkl', 'rb') as f:
        data = pickle.load(f)
        W = data['W']
        q_p_train = data['q_p_train']  # Ensure key matches training script
        epsilon = data['epsilon']
        kernel_name = data.get('kernel_name', 'gaussian')  # Default to 'gaussian' if not provided
    print("Global weight matrix and data loaded successfully.")

    kernel_func = rbf_kernels[kernel_name]
    print(f"Using kernel: {kernel_name}")
    print(f"Using epsilon: {epsilon}")

    U_p = np.load('pod_rbf_global_model/U_p.npy')
    U_s = np.load('pod_rbf_global_model/U_s.npy')
    U_full = np.hstack((U_p, U_s))  # Full U matrix with all modes
    print("POD basis matrices (U_p and U_s) loaded successfully.")

    with open('pod_rbf_global_model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Min-Max scaler loaded successfully.")

    # Additional parameters
    num_modes = U_p.shape[1] + U_s.shape[1]

    # Reconstruct the snapshot using global RBF interpolation (POD-RBF reconstruction)
    pod_rbf_reconstructed, reconstructed_q_s_rbf = reconstruct_snapshot_with_global_rbf(
        hdm_snap, U_p, U_s, q_p_train, W, scaler, epsilon, kernel_func, print_times=False
    )

    # Save the reconstructed data
    results_dir = "FOM_vs_POD-RBF_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Calculate and compare reconstruction errors
    pod_rbf_error = np.linalg.norm(hdm_snap - pod_rbf_reconstructed) / np.linalg.norm(hdm_snap)
    print(f"POD-RBF Reconstruction error: {pod_rbf_error:.6e}")

    # Define indices to plot (e.g., specific time steps)
    inds_to_plot = range(0, num_steps + 1, 50)  # Example: every 50 time steps

    # Prepare snapshots and labels to plot (fixed order):
    # 1. Simulation: HDM
    # 2. Reconstruction: POD-RBF (20+140)
    snaps_to_plot = [hdm_snap, pod_rbf_reconstructed]
    labels = ['Simulation: HDM', 'Reconstruction: POD-RBF (20+140)']
    colors = ['black', 'green']
    linewidths = [2, 2]

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
    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper right')
    plot_filename = f"plot_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}_n{num_modes}.png"
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")

    # ----- Compute per-time step relative errors and plot -----
    def compute_relative_errors(hdm, recon):
        # hdm and recon assumed to have shape (n, num_snapshots)
        num_snapshots = hdm.shape[1]
        errors = np.zeros(num_snapshots)
        for i in range(num_snapshots):
            norm_hdm = np.linalg.norm(hdm[:, i])
            if norm_hdm > 0:
                errors[i] = np.linalg.norm(hdm[:, i] - recon[:, i]) / norm_hdm
            else:
                errors[i] = 0.0
        return errors

    # Project the aggregated test snapshots onto the POD basis
    q_test = U_full.T @ hdm_snap
    # Split test projection into primary and secondary modes
    q_p_test_original = q_test[:20, :]
    q_s_test = q_test[20:150, :]
    err_pod_rbf = compute_relative_errors(hdm_snap, pod_rbf_reconstructed)
    err_pod_rbf_q_s = compute_relative_errors(q_s_test, reconstructed_q_s_rbf)

    plt.figure(figsize=(10, 6))
    time_steps = np.arange(hdm_snap.shape[1])
    plt.plot(time_steps, err_pod_rbf * 100, 'g-', linewidth=2, label='Reconstruction: POD-RBF (10+140)')
    plt.plot(time_steps, err_pod_rbf_q_s * 100, 'b-', linewidth=2, label='Reconstruction q_s: POD-RBF (10+140)')
    plt.xlabel('Time Step')
    plt.ylabel('Relative Error (\\%)')
    plt.title('Per-Time Step Relative Errors')
    plt.grid(True)
    plt.legend(loc='upper right')
    error_plot_filename = os.path.join(results_dir, f"per_time_step_errors_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}.png")
    plt.savefig(error_plot_filename, dpi=300)
    print(f"Per-time step error plot saved successfully to {error_plot_filename}")
    # Optionally, display the plot
    # plt.show()
