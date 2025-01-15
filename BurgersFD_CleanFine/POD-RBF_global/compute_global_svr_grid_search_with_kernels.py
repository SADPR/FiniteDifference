# compute_global_svr_with_kernels.py

import os
import numpy as np
import time
import pickle
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
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

def remove_duplicates(q_p, q_s, tolerance=1e-8):
    """
    Remove near-duplicate rows from q_p and the corresponding rows in q_s.
    
    Parameters:
    - q_p: Primary mode projection data (shape: primary_modes x total_snapshots).
    - q_s: Secondary mode projection data (shape: secondary_modes x total_snapshots).
    - tolerance: Tolerance for determining duplicates.

    Returns:
    - q_p_filtered: Filtered q_p with duplicates removed.
    - q_s_filtered: Corresponding filtered q_s with the same indices removed.
    """
    # Identify unique rows in q_p
    _, unique_indices = np.unique(np.round(q_p / tolerance) * tolerance, axis=1, return_index=True)
    # Sort indices to maintain order
    unique_indices = np.sort(unique_indices)
    
    # Filter q_p and q_s based on unique indices
    q_p_filtered = q_p[:, unique_indices]
    q_s_filtered = q_s[:, unique_indices]
    
    return q_p_filtered, q_s_filtered


def get_snapshot_params():
    """
    Generate a list of parameter vectors [mu1, mu2] within specified ranges.
    """
    MU1_RANGE = 4.25, 5.5
    MU2_RANGE = 0.015, 0.03
    SAMPLES_PER_MU = 3

    MU1_LOW, MU1_HIGH = MU1_RANGE
    MU2_LOW, MU2_HIGH = MU2_RANGE
    mu1_samples = np.linspace(MU1_LOW, MU1_HIGH, SAMPLES_PER_MU)
    mu2_samples = np.linspace(MU2_LOW, MU2_HIGH, SAMPLES_PER_MU)
    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples.append([mu1, mu2])
    return mu_samples


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

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Ensure the snapshot folder exists
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

    # Remove duplicates from q_p and synchronize with q_s
    print("Removing duplicates from q_p and synchronizing with q_s...")
    q_p, q_s = remove_duplicates(q_p, q_s)
    print(f"Duplicates removed. New shapes: q_p={q_p.shape}, q_s={q_s.shape}")

    # Normalize q_p using MinMaxScaler
    print("Normalizing q_p data using MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T
    print("Normalization complete.")

    # Save the scaler
    model_dir = "pod_svr_global_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Save the normalized q_p and q_s
    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q.npy'), q)
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q, q_p_normalized, q_s), saved successfully.")

    # Prepare training data
    q_p_train = q_p_normalized.T  # shape: (n_snapshots, n_primary_modes)
    q_s_train = q_s.T             # shape: (n_snapshots, n_secondary_modes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.2, random_state=42
    )

    # Import SVR from scikit-learn
    from sklearn.svm import SVR

    # We'll search over C, epsilon, and gamma each in logspace
    C_values = np.logspace(np.log10(1), np.log10(10), 2)        # [1, 10]
    eps_values = np.logspace(np.log10(0.001), np.log10(0.1), 5) # [0.001, 0.00316, 0.01, 0.0316, 0.1]   
    gamma_values = np.logspace(np.log10(1), np.log10(1000), 30) # [1, ..., 1000]

    # We define one kernel name or multiple (like 'rbf', 'linear'), but let's do only 'rbf'
    kernel_names = ['rbf']

    best_c = None
    best_eps = None
    best_gamma = None
    best_kernel = None
    lowest_error = np.inf

    print("Optimizing SVR with grid search over C, epsilon, gamma, and kernel...")

    from itertools import product
    for c_value, eps_value, gamma_val, kernel_name in product(C_values, eps_values, gamma_values, kernel_names):

        # We'll do dimension-by-dimension training for multi-output
        y_val_pred = np.zeros_like(y_val)
        for dim_idx in range(y_train.shape[1]):
            model_dim = SVR(kernel=kernel_name, C=c_value, gamma=gamma_val, epsilon=eps_value)
            model_dim.fit(X_train, y_train[:, dim_idx])
            y_val_pred_dim = model_dim.predict(X_val)
            y_val_pred[:, dim_idx] = y_val_pred_dim

        # Compute MSE across all dims
        error = mean_squared_error(y_val, y_val_pred)
        print(f"C={c_value:.5f}, eps={eps_value:.5f}, gamma={gamma_val:.5f}, kernel={kernel_name}, MSE={error:.5e}")

        if error < lowest_error:
            lowest_error = error
            best_c = c_value
            best_eps = eps_value
            best_gamma = gamma_val
            best_kernel = kernel_name

    if best_c is None:
        print("No suitable (C, epsilon, gamma, kernel) combination found. Exiting.")
        return

    print(f"Best combination found: C={best_c:.5f}, epsilon={best_eps:.5f}, gamma={best_gamma:.5f}, kernel={best_kernel}, MSE={lowest_error:.5e}")

    # Refit final multi-output model using all data
    final_models = []
    for dim_idx in range(q_s_train.shape[1]):
        final_svr = SVR(kernel=best_kernel, C=best_c, gamma=best_gamma, epsilon=best_eps)
        final_svr.fit(q_p_train, q_s_train[:, dim_idx])
        final_models.append(final_svr)

    # Save the "model" as 'svr_models'
    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'svr_models': final_models,  # List of SVR objects (one per dimension)
            'q_p_train': q_p_train,
            'q_s_train': q_s_train,
            'C_value': best_c,
            'epsilon_value': best_eps,
            'gamma_value': best_gamma,
            'kernel_name': best_kernel
        }, f)
    print(f"SVR-based model and data saved in {training_data_filename}.")

    print("Processing complete.")



if __name__ == '__main__':
    main()
