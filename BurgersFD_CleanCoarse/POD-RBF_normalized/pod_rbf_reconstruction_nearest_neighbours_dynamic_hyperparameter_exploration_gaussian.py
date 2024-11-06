import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure a non-interactive backend is used
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

# Define the Gaussian RBF kernel function
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def rbf_kernel(r, epsilon, kernel_type):
    """Selects the RBF kernel function based on the specified kernel type."""
    if kernel_type == "gaussian":
        return gaussian_rbf(r, epsilon)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

# Function to dynamically interpolate at new points using nearest neighbors
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors, kernel_type, print_times=False):
    """Interpolate at new points using nearest neighbors and solving the system on the fly."""
    start_time = time.time()
    dist, idx = kdtree.query(x_new, k=neighbors)
    kdtree_time = time.time()
    if print_times:
        print(f"KDTree query took: {kdtree_time - start_time:.6f} seconds")

    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)

    extract_time = time.time()
    if print_times:
        print(f"Extracting neighbors took: {extract_time - kdtree_time:.6f} seconds")

    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = rbf_kernel(dists_neighbors, epsilon, kernel_type)
    rbf_matrix_time = time.time()
    if print_times:
        print(f"RBF matrix computation took: {rbf_matrix_time - extract_time:.6f} seconds")

    Phi_neighbors += np.eye(neighbors) * 1e-8
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    solve_time = time.time()
    if print_times:
        print(f"Solving the linear system took: {solve_time - rbf_matrix_time:.6f} seconds")

    rbf_values = rbf_kernel(dist, epsilon, kernel_type)
    rbf_eval_time = time.time()
    if print_times:
        print(f"RBF evaluation for new point took: {rbf_eval_time - solve_time:.6f} seconds")

    f_new = rbf_values @ W_neighbors
    total_time = time.time() - start_time
    if print_times:
        print(f"Total interpolation process took: {total_time:.6f} seconds")

    return f_new

# Function to reconstruct a snapshot using the POD-RBF model with nearest neighbors and dynamic interpolation
def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, kernel_type, print_times=False):
    start_total_time = time.time()
    q = U_p.T @ snapshot
    q_p = q[:r, :]

    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        if print_times:
            print(f"Time step {i+1} of {q_p.shape[1]}")
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))

        q_s_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, kernel_type, print_times).T
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    #print(f"Total reconstruction process took: {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

# Function to find the best epsilon and neighbor values per snapshot
def find_best_gaussian_per_snapshot(epsilon_values, neighbor_values, hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, kernel_type='gaussian', print_times=False):
    num_snapshots = hdm_snap.shape[1]  # Number of snapshots
    best_reconstructed_snapshots = []
    best_parameters_per_snapshot = []
    overall_error = 0  # Track overall error

    # Loop over each time step (snapshot)
    for i in range(num_snapshots):
        snapshot = hdm_snap[:, i].reshape(-1, 1)  # Extract the i-th snapshot

        best_error = float('inf')
        best_reconstruction = None
        best_neighbors = None
        best_epsilon = None

        # Test each epsilon and neighbor combination for the current snapshot
        for epsilon in epsilon_values:
            for neighbors in neighbor_values:
                try:
                    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
                        snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, kernel_type, print_times
                    )
                    # Correctly reshape the reconstructed snapshot to match the original snapshot
                    reconstruction_error = np.linalg.norm(snapshot - pod_rbf_reconstructed.reshape(-1, 1)) / np.linalg.norm(snapshot)

                    # Keep track of the best reconstruction
                    if reconstruction_error < best_error:
                        best_error = reconstruction_error
                        best_reconstruction = pod_rbf_reconstructed
                        best_neighbors = neighbors
                        best_epsilon = epsilon

                except Exception as e:
                    print(f"Error with neighbors={neighbors} and epsilon={epsilon} for snapshot {i}: {e}")

        # Store the best reconstruction and parameters for this time step
        best_reconstructed_snapshots.append(best_reconstruction)
        best_parameters_per_snapshot.append({'neighbors': best_neighbors, 'epsilon': best_epsilon})
        overall_error += best_error
        print(f"Snapshot {i+1}: Best neighbors={best_neighbors}, Best epsilon={best_epsilon}, Best error={best_error:.6e}")

    best_reconstructed_snapshots = np.array(best_reconstructed_snapshots).squeeze().T

    return best_reconstructed_snapshots, best_parameters_per_snapshot, overall_error


# Main function
if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.75, 0.02]  # Example: mu1=4.56, mu2=0.019

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

    # Load the saved KDTree and training data (q_p and q_s)
    try:
        with open('modes/training_data.pkl', 'rb') as f:
            data = pickle.load(f)
            kdtree = data['KDTree']
            q_p_train = data['q_p']
            q_s_train = data['q_s']
            print("Loaded training data and KDTree successfully.")
    except FileNotFoundError:
        print("Training data file 'modes/training_data.pkl' not found.")
        exit(1)

    # Load U_p, U_s, and the full U matrix
    try:
        U_p = np.load('modes/U_p.npy')
        U_s = np.load('modes/U_s.npy')
        U_full = np.hstack((U_p, U_s))  # Full U matrix with 150 modes
        print("Loaded POD basis matrices (U_p and U_s) successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    r = 10  # Number of primary modes used
    kernel_type = "gaussian"  # Using the Gaussian kernel

    # Define ranges for epsilon and neighbor values
    epsilon_values = [0.1, 0.02, 0.01, 0.001, 0.0001]
    neighbor_values = [5, 10, 20, 50, 100, 500]

    # Run the exploration, finding the best epsilon and neighbors per snapshot
    best_reconstructed_snapshots, best_parameters_per_snapshot, overall_error = find_best_gaussian_per_snapshot(
        epsilon_values, neighbor_values, hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, kernel_type, print_times=False
    )

    # Save results
    results_dir = "Best_POD-RBF_Gaussian_Per_Snapshot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the best reconstructed snapshots
    pod_rbf_file_path = os.path.join(results_dir, "best_reconstructed_snapshots_pod_rbf_gaussian.npy")
    np.save(pod_rbf_file_path, best_reconstructed_snapshots)
    print(f"Best POD-RBF reconstructed snapshots saved to {pod_rbf_file_path}")

    # Save the best parameters (neighbors and epsilon) per snapshot
    best_parameters_file_path = os.path.join(results_dir, "best_parameters_per_snapshot.pkl")
    with open(best_parameters_file_path, 'wb') as f:
        pickle.dump(best_parameters_per_snapshot, f)
    print(f"Best parameters (neighbors and epsilon) per snapshot saved to {best_parameters_file_path}")

    # Calculate and print the overall reconstruction error
    pod_rbf_error = np.linalg.norm(hdm_snap - best_reconstructed_snapshots) / np.linalg.norm(hdm_snap)
    print(f"POD-RBF Reconstruction error (Gaussian kernel): {pod_rbf_error:.6e}")

    # Plot comparison of snapshots (HDM vs best POD-RBF)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    inds_to_plot = range(0, num_steps + 1, 20)  # Example: every 100 time steps
    snaps_to_plot = [hdm_snap, best_reconstructed_snapshots]
    labels = ['HDM', 'Best POD-RBF (Gaussian)']
    colors = ['black', 'green']
    linewidths = [2, 1]

    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(grid_x, grid_y, snap, inds_to_plot, label=label, fig_ax=(fig, ax1, ax2), color=color, linewidth=lw)

    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper right')
    plot_filename = os.path.join(results_dir, "comparison_plot_gaussian.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Comparison plot saved to {plot_filename}")

