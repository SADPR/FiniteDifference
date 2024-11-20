import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import sys

# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path if it's not already present
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now, import the required functions from hypernet2D
from hypernet2D import load_or_compute_snaps, make_2D_grid, plot_snaps

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(- (epsilon * r) ** 2)

def reconstruct_snapshot_with_pod_rbf_global(snapshot, U_p, U_s, q_p_train_normalized, W, scaler, epsilon, print_times=False):
    """
    Reconstruct snapshots using the POD-RBF model with global interpolation.

    Parameters:
    - snapshot: The high-dimensional snapshot to reconstruct (shape: (n_dofs, n_time_steps)).
    - U_p: Primary POD modes (shape: (n_dofs, n_p)).
    - U_s: Secondary POD modes (shape: (n_dofs, n_s)).
    - q_p_train_normalized: Normalized primary reduced coordinates from training data (shape: (n_p, M)).
    - W: Global weight matrix (shape: (n_s, M)).
    - scaler: The Min-Max scaler used for normalization.
    - epsilon: RBF hyperparameter.
    - print_times: Whether to print timing information.

    Returns:
    - reconstructed_snapshots: Reconstructed snapshots (shape: (n_dofs, n_time_steps)).
    """
    start_total_time = time.time()
    # Project the snapshot onto the primary modes
    q_p = U_p.T @ snapshot  # Shape: (n_p, n_time_steps)

    # Normalize q_p using the scaler
    q_p_normalized = scaler.transform(q_p.T).T  # Shape: (n_p, n_time_steps)

    # Compute pairwise distances between normalized q_p and q_p_train_normalized
    # Since q_p_train_normalized has shape (n_p, M), we need to compute distances for each time step
    # We'll compute the RBF values for each time step efficiently

    M = q_p_train_normalized.shape[1]  # Number of training data points
    n_time_steps = q_p.shape[1]

    reconstructed_snapshots = np.zeros((U_p.shape[0], n_time_steps))

    if print_times:
        print(f"Starting reconstruction of {n_time_steps} time steps using global RBF interpolation.")

    # Precompute norms of q_p_train_normalized for efficiency
    q_p_train_norms = np.sum(q_p_train_normalized ** 2, axis=0)  # Shape: (M,)

    for i in range(n_time_steps):
        start_time = time.time()
        q_p_i_normalized = q_p_normalized[:, i]  # Shape: (n_p,)

        # Compute squared distances between q_p_i_normalized and all training points
        # Efficiently compute: ||q_p_i_normalized - q_p_train_normalized||^2
        diff = q_p_train_normalized - q_p_i_normalized[:, np.newaxis]  # Shape: (n_p, M)
        squared_distances = np.sum(diff ** 2, axis=0)  # Shape: (M,)

        # Compute the RBF vector
        rbf_vector = gaussian_rbf(np.sqrt(squared_distances), epsilon)  # Shape: (M,)

        # Compute the secondary reduced coordinates
        q_s_i = W @ rbf_vector  # Shape: (n_s,)

        # Reconstruct the snapshot
        reconstructed_snapshot = U_p @ q_p[:, i] + U_s @ q_s_i  # Shape: (n_dofs,)
        reconstructed_snapshots[:, i] = reconstructed_snapshot

        if print_times:
            step_time = time.time() - start_time
            print(f"Time step {i+1}/{n_time_steps} reconstructed in {step_time:.6f} seconds.")

    total_time = time.time() - start_total_time
    if print_times:
        print(f"Total reconstruction process took: {total_time:.6f} seconds")

    return reconstructed_snapshots

# Function to reconstruct using standard POD
def reconstruct_snapshot_with_pod(snapshot, U, num_modes, print_times=False):
    """Reconstruct a snapshot using standard POD."""
    start_time = time.time()
    U_modes = U[:, :num_modes]
    q_pod = U_modes.T @ snapshot
    reconstructed_pod = U_modes @ q_pod
    if print_times:
        print(f"POD reconstruction took: {time.time() - start_time:.6f} seconds")
    return reconstructed_pod

if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.75, 0.02]  # Example: mu1=5.19, mu2=0.026

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

    # Load U_p, U_s, W, and the full U matrix
    try:
        U_p = np.load('modes/U_p.npy')
        U_s = np.load('modes/U_s.npy')
        U_full = np.hstack((U_p, U_s))  # Full U matrix with total modes
        print("Loaded POD basis matrices (U_p and U_s) successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Load the global weight matrix W
    try:
        W = np.load('modes/W.npy')
        print("Loaded global weight matrix W successfully.")
    except FileNotFoundError:
        print("Weight matrix file 'modes/W.npy' not found.")
        exit(1)

    # Load the saved scaler (Min-Max scaler)
    try:
        with open('modes/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded Min-Max scaler successfully.")
    except FileNotFoundError:
        print("Scaler file 'modes/scaler.pkl' not found.")
        exit(1)

    # Load normalized q_p training data
    try:
        q_p_train_normalized = np.load('modes/q_p_normalized.npy')  # Shape: (n_p, M)
        print("Loaded normalized q_p training data successfully.")
    except FileNotFoundError:
        print("File 'modes/q_p_normalized.npy' not found.")
        exit(1)

    # Load RBF hyperparameters
    try:
        with open('modes/rbf_params.pkl', 'rb') as f:
            rbf_params = pickle.load(f)
            epsilon = rbf_params['epsilon']
        print(f"Loaded RBF hyperparameters successfully (epsilon = {epsilon}).")
    except FileNotFoundError:
        print("RBF parameters file 'modes/rbf_params.pkl' not found.")
        exit(1)

    r = U_p.shape[1]  # Number of primary modes used
    num_modes = U_p.shape[1] + U_s.shape[1]  # Total number of modes

    # Boolean to control whether to perform POD comparison
    compare_pod = True  # Set to True to include Standard POD reconstruction in the plot
    # Boolean to control whether to print times for each step
    print_times = False

    # Reconstruct the snapshot using global RBF interpolation
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_global(
        hdm_snap, U_p, U_s, q_p_train_normalized, W, scaler, epsilon, print_times
    )

    # Initialize an empty variable for Standard POD reconstruction
    pod_reconstructed = None

    # Reconstruct the snapshot using standard POD with all modes if compare_pod is True
    if compare_pod:
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, U_full, num_modes, print_times)

    # Save the reconstructed data
    results_dir = "FOM_vs_POD-RBF_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    pod_rbf_file_path = os.path.join(results_dir, f"reconstructed_snapshot_pod_rbf_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy")
    #np.save(pod_rbf_file_path, pod_rbf_reconstructed)
    print(f"POD-RBF reconstructed snapshot saved successfully to {pod_rbf_file_path}")

    if compare_pod:
        pod_file_path = os.path.join(results_dir, f"reconstructed_snapshot_pod_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy")
        #np.save(pod_file_path, pod_reconstructed)
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

    # If Standard POD reconstruction is available, add it to the plot
    if compare_pod and pod_reconstructed is not None:
        snaps_to_plot.append(pod_reconstructed)
        labels.append('Standard POD')
        colors.append('blue')  # Assign a distinct color for Standard POD
        linewidths.append(2)

    # Plot the comparison using subplots for x and y slices
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(grid_x, grid_y, snap, inds_to_plot,
                   label=label,
                   fig_ax=(fig, ax1, ax2),
                   color=color,
                   linewidth=lw)

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
    #plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")

    # Optionally, display the plot
    plt.show()

