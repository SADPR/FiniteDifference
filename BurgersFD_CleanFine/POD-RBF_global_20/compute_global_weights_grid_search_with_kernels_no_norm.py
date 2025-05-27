# compute_global_weights_grid_search_with_kernels_no_norm.py

import os
import numpy as np
import time
import pickle
import logging
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

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

def get_snapshot_params():
    """
    Generate a list of parameter vectors [mu1, mu2] within specified ranges.
    """
    MU1_RANGE = (4.25, 5.5)
    MU2_RANGE = (0.015, 0.03)
    SAMPLES_PER_MU = 3

    mu1_samples = np.linspace(MU1_RANGE[0], MU1_RANGE[1], SAMPLES_PER_MU)
    mu2_samples = np.linspace(MU2_RANGE[0], MU2_RANGE[1], SAMPLES_PER_MU)
    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples.append([mu1, mu2])
    return mu_samples

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

def matern52_rbf(r, epsilon):
    """
    Matern 5/2 kernel.
    φ(r) = (1 + sqrt(5)*εr + 5(εr)^2 / 3) * exp(-sqrt(5)*εr)
    """
    sqrt5 = np.sqrt(5.0)
    scaled_r = epsilon * r
    tmp = sqrt5 * scaled_r
    return (1.0 + tmp + (tmp**2)/3.0) * np.exp(-tmp)

def compact_bump_rbf(r, epsilon):
    """
    Compactly-supported 'bump' function kernel.
    φ(r) = exp(1 / ((ε*r)^2 - 1)) if ε*r < 1, else 0
    """
    scaled_r = epsilon * r
    phi = np.zeros_like(scaled_r)
    inside = scaled_r < 1.0
    phi[inside] = np.exp(1.0 / (scaled_r[inside]**2 - 1.0))
    return phi

#############################
# Five Extra RBF Kernels
#############################

def thin_plate_spline_rbf(r, epsilon):
    """
    Thin Plate Spline (TPS).
    For 2D: φ(r) = (ε*r)^2 * log((ε*r) + 1e-15)
    (Adding a small constant inside log() to avoid log(0).)
    """
    scaled_r = epsilon * r
    # Avoid log(0) by adding a tiny offset
    scaled_r = np.where(scaled_r < 1e-15, 1e-15, scaled_r)
    return scaled_r**2 * np.log(scaled_r)

def wendland_c2_rbf(r, epsilon):
    """
    Wendland C2 kernel (compactly supported).
    φ(r) = (1 - ε*r)^4 * (4*ε*r + 1) for ε*r < 1, else 0
    """
    scaled_r = epsilon * r
    phi = np.zeros_like(scaled_r)
    inside = scaled_r < 1.0
    tmp = 1.0 - scaled_r[inside]
    phi[inside] = tmp**4 * (4.0 * scaled_r[inside] + 1.0)
    return phi

def rational_quadratic_rbf(r, epsilon, alpha=1.0):
    """
    Rational Quadratic kernel:
    φ(r) = (1 + (ε*r)^2 / (2*alpha))^(-alpha)
    For alpha=1, this approximates IMQ. For alpha->∞, approaches Gaussian.
    """
    scaled_r = (epsilon*r)**2
    return (1.0 + scaled_r / (2.0*alpha))**(-alpha)

def cubic_rbf(r, epsilon):
    """
    Cubic RBF:
    φ(r) = (ε*r)^3
    """
    return (epsilon*r)**3

def quintic_rbf(r, epsilon):
    """
    Quintic RBF:
    φ(r) = (ε*r)^5
    """
    return (epsilon*r)**5

#############################
# Combined Dictionary
#############################

rbf_kernels = {
    #'gaussian': gaussian_rbf
    'imq': inverse_multiquadric_rbf
    #'multiquadric': multiquadric_rbf
    #'linear': linear_rbf
    #'matern52': matern52_rbf,
    #'bump': compact_bump_rbf,
    #'thin_plate_spline': thin_plate_spline_rbf,
    #'wendland_c2': wendland_c2_rbf,
    #'rational_quadratic': rational_quadratic_rbf,
    #'cubic': cubic_rbf,
    #'quintic': quintic_rbf
}

def perform_pod(snaps, num_modes=150, method='rsvd', random_state=None):
    """
    Perform Proper Orthogonal Decomposition (POD) using SVD or Randomized SVD.
    """
    if method == 'rsvd':
        print("Performing Randomized SVD for POD...")
        start_time = time.time()
        from sklearn.utils.extmath import randomized_svd
        U, sigma, Vh = randomized_svd(snaps, n_components=num_modes, random_state=random_state)
        elapsed_time = time.time() - start_time
        print(f"Randomized SVD completed in {elapsed_time:.2f} seconds.")
    elif method == 'svd':
        print("Performing standard SVD for POD...")
        start_time = time.time()
        U, s, Vh = np.linalg.svd(snaps, full_matrices=False)
        U = U[:, :num_modes]
        sigma = s[:num_modes]
        elapsed_time = time.time() - start_time
        print(f"Standard SVD completed in {elapsed_time:.2f} seconds.")
    else:
        raise ValueError("Invalid POD method. Choose 'svd' or 'rsvd'.")

    basis = U
    return basis, sigma

def main():
    # Configure logging to capture missing snapshots
    logging.basicConfig(filename='missing_snapshots.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Use grid and initial conditions directly from config
    grid_x, grid_y = GRID_X, GRID_Y
    w0 = W0
    snap_folder = "../param_snaps"

    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        print("Please add the required snapshot files before running the script again.")
        return  # Exit since no snapshots are available

    # Generate all parameter samples
    mu_samples = get_snapshot_params()
    print(f"Total parameter samples: {len(mu_samples)}")

    # Attempt to load the shape of the first snapshot to determine snapshot dimensions
    try:
        first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
        snapshot_shape = first_snap.shape
        print(f"Shape of each snapshot: {snapshot_shape}")
    except FileNotFoundError as e:
        print(f"Error loading the first snapshot: {e}")
        print("Ensure that at least one snapshot exists to determine the snapshot dimensions.")
        logging.error(f"Error loading the first snapshot: {e}")
        return  # Exit since snapshot dimensions are unknown

    snap_count = len(mu_samples)  # Total number of parameter combinations
    total_snaps = snapshot_shape[1] * snap_count  # Total number of snapshots (time steps * parameter combinations)
    print(f"Total number of snapshots to aggregate: {total_snaps}")

    # Pre-allocate memory for all snapshots
    snaps = np.zeros((snapshot_shape[0], total_snaps))

    # Collect snapshots into the pre-allocated array
    col_offset = 0
    successful_mu = []
    missing_mu = []

    for idx, mu in enumerate(mu_samples):
        try:
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
            snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu  # Insert directly
            col_offset += snap_mu.shape[1]  # Update column offset for the next parameter set
            successful_mu.append(mu)
            print(f"Loaded snapshot {idx + 1}/{snap_count} for mu1={mu[0]}, mu2={mu[1]}")
        except FileNotFoundError as e:
            print(e)
            missing_mu.append(mu)
            logging.info(f"Missing snapshots for mu1={mu[0]}, mu2={mu[1]}")

    # Trim the pre-allocated array in case some snapshots are missing
    if missing_mu:
        loaded_snaps = col_offset
        snaps = snaps[:, :loaded_snaps]
    else:
        loaded_snaps = total_snaps  # All snapshots loaded

    print(f"Successfully loaded {loaded_snaps} snapshots out of {total_snaps}.")

    if missing_mu:
        print("Missing parameter sets have been logged in 'missing_snapshots.log'.")

    if snaps.size == 0:
        print("No snapshots were loaded. Exiting the workflow.")
        return

    print(f"Combined snapshot matrix shape: {snaps.shape}")

    # Define whether to compute the basis or load a precomputed one
    compute_basis = False  # Set to False to load a precomputed basis

    if not compute_basis:
        # Load a precomputed basis
        basis_path = 'basis.npy'
        if os.path.exists(basis_path):
            basis = np.load(basis_path, allow_pickle=True)
            print(f"Loaded precomputed basis from {basis_path}.")
        else:
            print(f"Basis file '{basis_path}' not found. Please compute the basis first.")
            return
    else:
        # Define POD parameters
        pod_method = 'rsvd'  # Choose between 'svd' or 'rsvd'
        num_modes = 150  # Total number of POD modes to retain
        random_state = 42  # For reproducibility (only used if pod_method='rsvd')

        # Perform POD to compute the Reduced Order Basis (ROB)
        basis, sigma = perform_pod(snaps, num_modes=num_modes, method=pod_method, random_state=random_state)
        print(f"Computed POD basis with method '{pod_method}' and {num_modes} modes.")

        # Save the computed basis and singular values for future use
        np.save('basis.npy', basis)
        np.save('sigma.npy', sigma)
        print("Saved computed basis to 'basis.npy' and singular values to 'sigma.npy'.")

    # Define how many primary modes to use
    primary_modes = 10
    total_modes = 150  # Ensure total_modes >= primary_modes

    # Project the snapshots onto the POD basis
    print("Projecting snapshots onto the POD basis...")
    projection_start_time = time.time()
    q = basis.T @ snaps  # Project snapshots onto the POD basis
    q_p = q[:primary_modes, :]  # Primary mode projections
    q_s = q[primary_modes:total_modes, :]  # Secondary mode projections
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")
    del snaps

    # ------------------------------------------------------------------------------------
    # NO NORMALIZATION Step -- we skip MinMaxScaler or any other scaling
    # We'll just treat q_p as-is
    # ------------------------------------------------------------------------------------

    model_dir = "pod_rbf_global_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the q_p and q_s for future use
    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q.npy'), q)
    np.save(os.path.join(model_dir, 'q_p.npy'), q_p)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q, q_p, q_s), saved successfully.")

    # Prepare training data
    q_p_train = q_p.T  # shape: (num_snapshots, num_primary_modes)
    q_s_train = q_s.T  # shape: (num_snapshots, num_secondary_modes)

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.2, random_state=42
    )

    # ### ADDED: Create a dummy dist array of zeros, then filter near-duplicates in X_train
    dummy_dist = np.zeros(X_train.shape[0])
    X_train_f, y_train_f, dist_f = filter_similar_neighbors(
        X_train, y_train, dummy_dist, similarity_threshold=1e-2
    )
    print(f"Filtered out near-duplicates in training set: {X_train.shape[0]} -> {X_train_f.shape[0]}")

    # Reassign
    X_train = X_train_f
    y_train = y_train_f
    # ### END ADD

    # Define grids for epsilon and kernel names
    epsilon_values = np.logspace(np.log10(0.001), np.log10(10), 20)
    kernel_names = list(rbf_kernels.keys())

    best_epsilon = None
    best_kernel_name = None
    lowest_error = np.inf
    best_W = None

    print("Optimizing epsilon and kernel using grid search...")
    for epsilon, kernel_name in product(epsilon_values, kernel_names):
        kernel_func = rbf_kernels[kernel_name]

        # Compute distance matrix between training points
        dists_train = np.linalg.norm(X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2)
        Phi_train = kernel_func(dists_train, epsilon)
        Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

        try:
            U, S, Vt = np.linalg.svd(Phi_train, full_matrices=False)
            tol = 1e-8
            S_inv = np.diag([1/s if s>tol else 0 for s in S])
            W = Vt.T @ S_inv @ (U.T @ y_train)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}. Skipping.")
            continue

        # Compute distance matrix for validation set
        dists_val = np.linalg.norm(X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2)
        Phi_val = kernel_func(dists_val, epsilon)

        y_val_pred = Phi_val @ W
        error = mean_squared_error(y_val, y_val_pred)
        print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, Validation MSE: {error:.5e}")

        if error < lowest_error:
            lowest_error = error
            best_epsilon = epsilon
            best_kernel_name = kernel_name
            best_W = W.copy()

    if best_epsilon is None or best_kernel_name is None:
        print("No suitable epsilon and kernel combination found. Exiting.")
        return

    print(f"Best epsilon found: {best_epsilon:.5f}")
    print(f"Best kernel found: {best_kernel_name} with MSE: {lowest_error:.5e}")

    # Use the best epsilon and kernel to compute final W using all data
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]

    dists_train_all = np.linalg.norm(q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2)
    Phi_train_all = kernel_func(dists_train_all, epsilon)
    Phi_train_all += np.eye(Phi_train_all.shape[0]) * 1e-8

    U, S, Vt = np.linalg.svd(Phi_train_all, full_matrices=False)
    tol = 1e-8
    S_inv = np.diag([1/s if s>tol else 0 for s in S])
    W_final = Vt.T @ S_inv @ (U.T @ q_s_train)

    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W_final,
            'q_p_train': q_p_train,   # unscaled primary coords
            'q_s_train': q_s_train,   # secondary coords
            'epsilon': epsilon,
            'kernel_name': best_kernel_name
        }, f)
    print(f"Global weight matrix and data saved in {training_data_filename}.")

    print("Processing complete (no normalization).")


if __name__ == '__main__':
    main()

