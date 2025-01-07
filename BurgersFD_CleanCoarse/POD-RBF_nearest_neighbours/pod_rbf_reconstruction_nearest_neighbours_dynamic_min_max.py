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
from hypernet2D import load_or_compute_snaps, plot_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

# Function to compute the Gaussian RBF kernel
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    """Linear RBF kernel function."""
    return r

def filter_similar_neighbors(X_neighbors, Y_neighbors, dist, similarity_threshold=1e-4):
    """
    Filters out neighbors that are too close to each other based on a similarity threshold.
    
    Parameters:
    - X_neighbors: Array of nearest neighbor coordinates (n_neighbors, n_features).
    - Y_neighbors: Array of secondary modes for the neighbors (n_neighbors, n_output_dims).
    - dist: Array of distances to the target point (1, n_neighbors) or (n_neighbors,).
    - similarity_threshold: Threshold distance below which points are considered too close.
    
    Returns:
    - X_filtered, Y_filtered, dist_filtered: Filtered arrays with similar neighbors removed.
    """
    # Ensure `dist` is a 1D array to match `mask`
    dist = dist.flatten()
    n_neighbors = X_neighbors.shape[0]
    mask = np.ones(n_neighbors, dtype=bool)  # Start with all neighbors included
    
    for i in range(n_neighbors):
        if mask[i]:  # Only check if this neighbor is still included
            # Identify close neighbors and exclude them
            too_close = np.linalg.norm(X_neighbors - X_neighbors[i], axis=1) < similarity_threshold
            too_close[i] = False  # Keep the current point
            mask[too_close] = False  # Exclude other close points
            
    # Apply the mask
    X_filtered = X_neighbors[mask]
    Y_filtered = Y_neighbors[mask]
    dist_filtered = dist[mask]
    
    return X_filtered, Y_filtered, dist_filtered

# Function to dynamically interpolate at new points using nearest neighbors
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors, print_times=False):
    """Interpolate at new points using nearest neighbors and solving the system on the fly."""
    start_time = time.time()
    dist, idx = kdtree.query(x_new, k=neighbors)

    kdtree_time = time.time()
    if print_times:
        print(f"KDTree query took: {kdtree_time - start_time:.6f} seconds")

    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx, :].reshape(neighbors, -1)

    '''
    # Apply the filtering function
    similarity_threshold = 0.001  # Define this threshold based on your data distribution
    X_neighbors_filtered, Y_neighbors_filtered, dist_filtered = filter_similar_neighbors(X_neighbors, Y_neighbors, dist, similarity_threshold)

    # Proceed with filtered neighbors
    dists_neighbors = np.linalg.norm(X_neighbors_filtered[:, np.newaxis] - X_neighbors_filtered[np.newaxis, :], axis=-1)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    Phi_neighbors += np.eye(len(X_neighbors_filtered)) * 1e-8
    cond_number = np.linalg.cond(Phi_neighbors)
    print(f"Condition number of Phi: {cond_number}")
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors_filtered)
    '''

    extract_time = time.time()
    if print_times:
        print(f"Extracting neighbors took: {extract_time - kdtree_time:.6f} seconds")

    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    rbf_matrix_time = time.time()
    if print_times:
        print(f"RBF matrix computation took: {rbf_matrix_time - extract_time:.6f} seconds")

    Phi_neighbors += np.eye(neighbors) * 1e-8
    cond_number = np.linalg.cond(Phi_neighbors)
    print(f"Condition number of Phi: {cond_number}")
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    solve_time = time.time()

    if print_times:
        print(f"Solving the linear system took: {solve_time - rbf_matrix_time:.6f} seconds")

    rbf_values = gaussian_rbf(dist, epsilon)
    rbf_eval_time = time.time()
    if print_times:
        print(f"RBF evaluation for new point took: {rbf_eval_time - solve_time:.6f} seconds")

    f_new = rbf_values @ W_neighbors
    total_time = time.time() - start_time
    if print_times:
        print(f"Total interpolation process took: {total_time:.6f} seconds")

    return f_new

# Function to reconstruct a snapshot using the POD-RBF model with nearest neighbors and dynamic interpolation
def reconstruct_snapshot_with_pod_rbf_neighbors(snapshot, U_p, U_s, q_p_train, q_s_train, kdtree, scaler, r, epsilon, neighbors, print_times=False):
    start_total_time = time.time()
    q = U_p.T @ snapshot
    q_p = q[:r, :]

    reconstructed_snapshots_rbf = []
    for i in range(q_p.shape[1]):
        if print_times:
            print(f"Time step {i+1} of {q_p.shape[1]}")
        q_p_sample = np.array(q_p[:, i].reshape(1, -1))

        # Normalize q_p_sample using the scaler (Min-Max normalization)
        q_p_sample_normalized = scaler.transform(q_p_sample)

        q_s_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, q_p_sample_normalized, epsilon, neighbors, print_times).T
        reconstructed_snapshot_rbf = U_p @ q_p[:, i] + U_s @ q_s_pred.reshape(-1)
        reconstructed_snapshots_rbf.append(reconstructed_snapshot_rbf)

    reconstructed_snapshots_rbf = np.array(reconstructed_snapshots_rbf).squeeze().T
    print(f"Total reconstruction process took: {time.time() - start_total_time:.6f} seconds")

    return reconstructed_snapshots_rbf

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

# Function to compare and plot snapshots
def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
    """
    Compare multiple snapshots by plotting them at specified indices.

    Parameters:
    - snaps_to_plot: List of snapshot arrays to plot. Each array should have shape (total_dofs, total_snapshots).
    - inds_to_plot: Iterable of snapshot indices to plot.
    - labels: List of labels for each snapshot array.
    - colors: List of colors for each snapshot array.
    - linewidths: List of linewidths for each snapshot array.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        for idx in inds_to_plot:
            if idx < snap.shape[1]:
                # Plot only the first component or a specific component as needed
                plt.plot(snap[:, idx], label=f"{label} - Snapshot {idx}" if idx == inds_to_plot[0] else "", 
                         color=color, linewidth=lw)
    plt.xlabel('Dof Index')
    plt.ylabel('Amplitude')
    plt.title('Comparison of HDM, POD, and POD-RBF Reconstructed Snapshots')
    plt.legend(loc='upper right')
    plt.grid(True)

if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [5.19, 0.026]  # Example: mu1=5.19, mu2=0.026

    # Define simulation parameters
    # Use grid and initial conditions directly from config
    grid_x, grid_y = GRID_X, GRID_Y
    w0 = W0

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
        hdm_snap = load_or_compute_snaps(target_mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
        print(f"Loaded HDM snapshot for mu1={target_mu[0]}, mu2={target_mu[1]}")
    except FileNotFoundError as e:
        print(e)
        exit(1)  # Exit since the target snapshot is not available

    # Load the saved KDTree and training data (q_p and q_s)
    try:
        with open('pod_rbf_nearest_model/training_data.pkl', 'rb') as f:
            data = pickle.load(f)
            kdtree = data['KDTree']
            q_p_train = data['q_p']
            q_s_train = data['q_s']
        print("Loaded training data and KDTree successfully.")
    except FileNotFoundError:
        print("Training data file 'pod_rbf_nearest_model/training_data.pkl' not found.")
        exit(1)

    # Load U_p, U_s, and the full U matrix
    try:
        U_p = np.load('pod_rbf_nearest_model/U_p.npy')
        U_s = np.load('pod_rbf_nearest_model/U_s.npy')
        U_full = np.hstack((U_p, U_s))  # Full U matrix with 150 modes
        print("Loaded POD basis matrices (U_p and U_s) successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Load the saved scaler (Min-Max scaler)
    try:
        with open('pod_rbf_nearest_model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded Min-Max scaler successfully.")
    except FileNotFoundError:
        print("Scaler file 'pod_rbf_nearest_model/scaler.pkl' not found.")
        exit(1)


    epsilon = 10.0
    neighbors = 5
    r = 10  # Number of primary modes used
    num_modes = 150

    # Boolean to control whether to perform POD comparison
    compare_pod = True  # Set to True to include Standard POD reconstruction in the plot
    # Boolean to control whether to print times for each step
    print_times = False

    # Reconstruct the snapshot using dynamic RBF interpolation with nearest neighbors
    pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
        hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, scaler, r, epsilon, neighbors, print_times
    )

    # Initialize an empty variable for Standard POD reconstruction
    pod_reconstructed = None

    # Reconstruct the snapshot using standard POD with 150 modes if compare_pod is True
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

    # Define indices to plot (e.g., specific time steps or spatial indices)
    inds_to_plot = range(0, NUM_STEPS + 1, 100)  # Example: every 100 time steps

    # Prepare snapshots to plot
    snaps_to_plot = [hdm_snap, pod_rbf_reconstructed]
    labels = ['HDM', 'POD-RBF']
    colors = ['black', 'green']
    linewidths = [2, 1]

    # If Standard POD reconstruction is available, add it to the plot
    if compare_pod and pod_reconstructed is not None:
        snaps_to_plot.append(pod_reconstructed)
        labels.append('Standard POD')
        colors.append('blue')  # Assign a distinct color for Standard POD
        linewidths.append(1)

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
    plot_filename = f"plot_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}_n{num_modes}_neighbors{neighbors}.png"
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")

    # Optionally, display the plot
    # plt.show()
