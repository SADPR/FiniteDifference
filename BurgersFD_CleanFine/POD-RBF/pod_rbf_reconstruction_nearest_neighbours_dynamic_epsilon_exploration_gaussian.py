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

# Function to calculate epsilon using the inverse of average nearest-neighbor distance for specific neighbors
def calculate_epsilon_inverse_nearest_neighbor(kdtree, q_p_train, x_new, neighbors):
    dist, idx = kdtree.query(x_new, k=neighbors)  # Find nearest neighbors for x_new
    average_distance = np.mean(dist)  # Calculate the average distance to the neighbors
    epsilon = 1 / average_distance  # Inverse of average distance
    return epsilon

# Function to calculate epsilon proportional to the range of distances for specific neighbors
def calculate_epsilon_proportional_to_domain_size(kdtree, q_p_train, x_new, neighbors, constant=1.0):
    dist, idx = kdtree.query(x_new, k=neighbors)  # Find nearest neighbors for x_new
    domain_size = np.max(dist) - np.min(dist)  # Calculate the range of distances between neighbors
    epsilon = constant / domain_size  # Proportional to domain size (distance range)
    return epsilon

# Define the Gaussian RBF kernel function
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

# Function to dynamically interpolate at new points using nearest neighbors
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors):
    dist, idx = kdtree.query(x_new, k=neighbors)
    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)

    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = np.exp(-(epsilon * dists_neighbors) ** 2)  # Gaussian RBF kernel
    Phi_neighbors += np.eye(neighbors) * 1e-8
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)

    rbf_values = np.exp(-(epsilon * dist) ** 2)  # RBF for new point
    f_new = rbf_values @ W_neighbors
    return f_new

# Function to find the best epsilon and neighbor values per snapshot
def find_best_epsilon_per_snapshot(hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, constant=0.1, neighbors=40):
    num_snapshots = hdm_snap.shape[1]  # Number of snapshots
    best_reconstructed_snapshots = []
    best_parameters_per_snapshot = []
    overall_error = 0  # Track overall error

    # Loop over each time step (snapshot)
    for i in range(num_snapshots):
        snapshot = hdm_snap[:, i].reshape(-1, 1)  # Extract the i-th snapshot
        best_error = float('inf')
        best_reconstruction = None
        best_epsilon = None

        # Initialize reconstruction for this snapshot
        reconstructed_snapshots_rbf = []

        q = U_p.T @ snapshot
        q_p = q[:r, :]  # Reduced coordinates

        # Loop through each reduced coordinate sample (q_p)
        for j in range(q_p.shape[1]):
            q_p_sample = np.array(q_p[:, j].reshape(1, -1))

            # Calculate epsilon dynamically for this specific point's neighbors
            epsilon_inv = 0.01#calculate_epsilon_inverse_nearest_neighbor(kdtree, q_p_train, q_p_sample, neighbors)
            epsilon_domain = 0.01#calculate_epsilon_proportional_to_domain_size(kdtree, q_p_train, q_p_sample, neighbors, constant)

            # Test with epsilon from inverse nearest-neighbor distance
            q_s_pred_inv = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample, epsilon_inv, neighbors)
            recon_inv = U_p @ q_p[:, j] + U_s @ q_s_pred_inv.reshape(-1)
            error_inv = np.linalg.norm(snapshot - recon_inv.reshape(-1, 1)) / np.linalg.norm(snapshot)

            # Test with epsilon proportional to domain size
            q_s_pred_domain = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample, epsilon_domain, neighbors)
            recon_domain = U_p @ q_p[:, j] + U_s @ q_s_pred_domain.reshape(-1)
            error_domain = np.linalg.norm(snapshot - recon_domain.reshape(-1, 1)) / np.linalg.norm(snapshot)

            # Choose the best reconstruction and epsilon for this point
            if error_inv < error_domain:
                reconstructed_snapshots_rbf.append(recon_inv)
                best_epsilon = epsilon_domain
            else:
                reconstructed_snapshots_rbf.append(recon_domain)
                best_epsilon = epsilon_inv

        # Convert list of reconstructions to array
        best_reconstruction = np.array(reconstructed_snapshots_rbf).squeeze().T
        best_error = np.linalg.norm(snapshot - best_reconstruction.reshape(-1,1)) / np.linalg.norm(snapshot)

        # Store the best reconstruction and parameters for this time step
        best_reconstructed_snapshots.append(best_reconstruction)
        best_parameters_per_snapshot.append({'neighbors': neighbors, 'epsilon': best_epsilon})
        overall_error += best_error
        print(f"Snapshot {i+1}: Best error={best_error:.6e}, neighbors: {neighbors}, epsilon: {best_epsilon}")

    best_reconstructed_snapshots = np.array(best_reconstructed_snapshots).squeeze().T
    return best_reconstructed_snapshots, best_parameters_per_snapshot, overall_error

# Main function
if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.75, 0.02]  # Example: mu1=4.56, mu2=0.019

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
        exit(1)

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
    constant = 1.0  # Constant for domain-size proportional epsilon

    # Run the exploration, finding the best epsilon and neighbors per snapshot
    best_reconstructed_snapshots, best_parameters_per_snapshot, overall_error = find_best_epsilon_per_snapshot(
        hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, constant
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

    '''
    # Plot comparison of snapshots (HDM vs best POD-RBF)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    inds_to_plot = range(0, num_steps + 1, 20)
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
    '''
