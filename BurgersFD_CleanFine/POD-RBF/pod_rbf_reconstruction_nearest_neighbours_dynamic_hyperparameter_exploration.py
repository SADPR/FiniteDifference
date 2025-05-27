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

# Define various RBF kernels
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def multiquadric_rbf(r, epsilon):
    return np.sqrt(1 + (epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    return r

def cubic_rbf(r, epsilon):
    return r ** 3

def thin_plate_spline_rbf(r, epsilon):
    return r ** 2 * np.log(r + np.finfo(float).eps)

def power_rbf(r, epsilon, p=2):
    return r ** p

def exponential_rbf(r, epsilon):
    return np.exp(-epsilon * r)

def polyharmonic_spline_rbf(r, epsilon):
    return r ** 2 * np.log(r + np.finfo(float).eps)

def rbf_kernel(r, epsilon, kernel_type):
    """Selects the RBF kernel function based on the specified kernel type."""
    if kernel_type == "gaussian":
        return gaussian_rbf(r, epsilon)
    elif kernel_type == "multiquadric":
        return multiquadric_rbf(r, epsilon)
    elif kernel_type == "inverse_multiquadric":
        return inverse_multiquadric_rbf(r, epsilon)
    elif kernel_type == "linear":
        return linear_rbf(r, epsilon)
    elif kernel_type == "cubic":
        return cubic_rbf(r, epsilon)
    elif kernel_type == "thin_plate_spline":
        return thin_plate_spline_rbf(r, epsilon)
    elif kernel_type == "power":
        return power_rbf(r, epsilon, p=2)
    elif kernel_type == "exponential":
        return exponential_rbf(r, epsilon)
    elif kernel_type == "polyharmonic_spline":
        return polyharmonic_spline_rbf(r, epsilon)
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

# Function to explore different hyperparameter combinations
def explore_combinations(epsilon_values, neighbor_values, kernel_types, hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, num_modes, compare_pod=False, print_times=False):
    results_dir = "FOM_vs_POD-RBF_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for kernel in kernel_types:
        for epsilon in epsilon_values:
            for neighbors in neighbor_values:
                #print(f"Running for kernel: {kernel}, epsilon: {epsilon}, neighbors: {neighbors}")

                # Reconstruct the snapshot using dynamic RBF interpolation with the current kernel, epsilon, and neighbors
                pod_rbf_reconstructed = reconstruct_snapshot_with_pod_rbf_neighbors(
                    hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, epsilon, neighbors, kernel, print_times
                )

                # Save the reconstructed data for this combination
                pod_rbf_file_path = os.path.join(
                    results_dir,
                    f"reconstructed_snapshot_pod_rbf_{kernel}_epsilon_{epsilon}_neighbors_{neighbors}_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
                )
                #np.save(pod_rbf_file_path, pod_rbf_reconstructed)
                #print(f"POD-RBF reconstructed snapshot saved for {kernel}, epsilon={epsilon}, neighbors={neighbors} to {pod_rbf_file_path}")

                # If compare_pod is True, perform Standard POD reconstruction and comparison
                if compare_pod:
                    pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, U_full, num_modes, print_times)
                    pod_file_path = os.path.join(
                        results_dir,
                        f"reconstructed_snapshot_pod_{kernel}_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
                    )
                    #np.save(pod_file_path, pod_reconstructed)
                    #print(f"Standard POD reconstructed snapshot saved to {pod_file_path}")

                # Calculate and compare reconstruction errors
                pod_rbf_error = np.linalg.norm(hdm_snap - pod_rbf_reconstructed) / np.linalg.norm(hdm_snap)
                print(f"{kernel} Kernel | epsilon={epsilon}, neighbors={neighbors}: POD-RBF Reconstruction error: {pod_rbf_error:.6e}")

                # Optionally compare POD error if available
                if compare_pod and pod_reconstructed is not None:
                    pod_error = np.linalg.norm(hdm_snap - pod_reconstructed) / np.linalg.norm(hdm_snap)
                    #print(f"Standard POD Reconstruction error: {pod_error:.6e}")
                
                '''
                # Plot the comparison using subplots for x and y slices
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                # Define indices to plot (e.g., specific time steps or spatial indices)
                inds_to_plot = range(0, num_steps + 1, 100)  # Example: every 100 time steps

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

                # Plot each snapshot
                for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
                    plot_snaps(grid_x, grid_y, snap, inds_to_plot,
                            label=label,
                            fig_ax=(fig, ax1, ax2),
                            color=color,
                            linewidth=lw)

                # Finalize and save the plot
                plt.tight_layout()
                plt.grid(True)
                plt.legend(loc='upper right')
                plot_filename = f"plot_{kernel}_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]:.3f}_epsilon{epsilon}_neighbors{neighbors}.png"
                plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
                plt.close()
                #print(f"Comparison plot saved to {os.path.join(results_dir, plot_filename)}")
                '''

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
    num_modes = 150
    compare_pod = True  # Set to True to include Standard POD reconstruction in the plot

    # Define ranges for RBF hyperparameters
    epsilon_values = [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]
    neighbor_values = [5, 10, 20, 50, 100, 200]#[5, 10, 20, 50, 100, 500]
    #kernel_types = [
    #    "linear","gaussian", "multiquadric", "inverse_multiquadric",
    #    "cubic", "thin_plate_spline", "power", "exponential", "polyharmonic_spline"
    #]
    kernel_types = ["gaussian","inverse_multiquadric","linear"]

    # Run the exploration
    explore_combinations(epsilon_values, neighbor_values, kernel_types, hdm_snap, U_p, U_s, q_p_train, q_s_train, kdtree, r, num_modes, compare_pod=compare_pod, print_times=False)


