# compute_global_weights_grid_search_with_kernels.py

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
    #'linear': linear_rbf
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
    # Configure logging
    logging.basicConfig(filename='missing_snapshots.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    grid_x, grid_y = GRID_X, GRID_Y
    w0 = W0

    snap_folder = "../param_snaps"

    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        print("Please add the required snapshot files before running the script again.")
        return

    mu_samples = get_snapshot_params()
    print(f"Total parameter samples: {len(mu_samples)}")

    try:
        first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
        snapshot_shape = first_snap.shape
        print(f"Shape of each snapshot: {snapshot_shape}")
    except FileNotFoundError as e:
        print(f"Error loading the first snapshot: {e}")
        logging.error(f"Error loading the first snapshot: {e}")
        return

    snap_count = len(mu_samples)
    total_snaps = snapshot_shape[1] * snap_count
    print(f"Total number of snapshots to aggregate: {total_snaps}")

    snaps = np.zeros((snapshot_shape[0], total_snaps))

    col_offset = 0
    successful_mu = []
    missing_mu = []

    for idx, mu in enumerate(mu_samples):
        try:
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
            snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu
            col_offset += snap_mu.shape[1]
            successful_mu.append(mu)
            print(f"Loaded snapshot {idx + 1}/{snap_count} for mu1={mu[0]}, mu2={mu[1]}")
        except FileNotFoundError as e:
            print(e)
            missing_mu.append(mu)
            logging.info(f"Missing snapshots for mu1={mu[0]}, mu2={mu[1]}")

    if missing_mu:
        loaded_snaps = col_offset
        snaps = snaps[:, :loaded_snaps]
    else:
        loaded_snaps = total_snaps

    print(f"Successfully loaded {loaded_snaps} snapshots out of {total_snaps}.")
    if missing_mu:
        print("Missing parameter sets logged.")

    if snaps.size == 0:
        print("No snapshots were loaded. Exiting.")
        return

    print(f"Combined snapshot matrix shape: {snaps.shape}")

    compute_basis = False

    if not compute_basis:
        basis_path = 'basis.npy'
        if os.path.exists(basis_path):
            basis = np.load(basis_path, allow_pickle=True)
            print(f"Loaded precomputed basis from {basis_path}.")
        else:
            print(f"Basis file '{basis_path}' not found. Please compute the basis first.")
            return
    else:
        pod_method = 'rsvd'
        num_modes = 150
        random_state = 42
        basis, sigma = perform_pod(snaps, num_modes=num_modes, method=pod_method, random_state=random_state)
        np.save('basis.npy', basis)
        np.save('sigma.npy', sigma)
        print("Computed and saved basis and sigma.")

    primary_modes = 10
    total_modes = 150

    print("Projecting snapshots onto the POD basis...")
    projection_start_time = time.time()
    q = basis.T @ snaps
    q_p = q[:primary_modes, :]
    q_s = q[primary_modes:total_modes, :]
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")
    del snaps

    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T
    print("Normalization complete.")

    model_dir = "pod_rbf_global_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q.npy'), q)
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Saved q_p_normalized and q_s.")

    q_p_train = q_p_normalized.T
    q_s_train = q_s.T

    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.1, random_state=42
    )

    # Load U_p and U_s for full-field reconstruction
    U_p = np.load(os.path.join(model_dir, 'U_p.npy'))
    U_s = np.load(os.path.join(model_dir, 'U_s.npy'))

    epsilon_values = np.logspace(np.log10(1), np.log10(10), 10)
    kernel_names = list(rbf_kernels.keys())

    best_epsilon = None
    best_kernel_name = None
    lowest_error = np.inf
    min_max_pointwise_error = np.inf
    best_W = None

    print("Optimizing epsilon and kernel using grid search...")
    # For full-field checks: select a small subset of validation samples
    # to keep computation manageable.
    max_samples_for_full_field = 5
    val_indices_subset = np.random.choice(X_val.shape[0], size=min(max_samples_for_full_field, X_val.shape[0]), replace=False)

    for epsilon, kernel_name in product(epsilon_values, kernel_names):
        kernel_func = rbf_kernels[kernel_name]

        dists_train = np.linalg.norm(
            X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        Phi_train = kernel_func(dists_train, epsilon)
        Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

        try:
            U_, S_, Vt_ = np.linalg.svd(Phi_train, full_matrices=False)
            tolerance = 1e-8
            S_inv = np.diag([1/s if s > tolerance else 0 for s in S_])
            W = Vt_.T @ S_inv @ U_.T @ y_train
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}. Skipping.")
            continue

        dists_val = np.linalg.norm(
            X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        Phi_val = kernel_func(dists_val, epsilon)
        y_val_pred = Phi_val @ W

        # Compute a small subset check for max pointwise error
        q_p_val_norm = X_val[val_indices_subset, :]
        q_p_val_original = scaler.inverse_transform(q_p_val_norm)
        q_s_val_pred_subset = y_val_pred[val_indices_subset, :]
        q_s_val_subset = y_val[val_indices_subset, :]

        w_pred = U_p @ q_p_val_original.T + U_s @ q_s_val_pred_subset.T
        w_ref = U_p @ q_p_val_original.T + U_s @ q_s_val_subset.T

        max_abs_error = np.max(np.abs(w_ref - w_pred))
        # MSE just for info
        error_mse = mean_squared_error(y_val, y_val_pred)

        print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, MSE: {error_mse:.5e}, Max Abs Err: {max_abs_error:.5e}")

        # Update if we found a lower max pointwise error
        if max_abs_error < min_max_pointwise_error:
            min_max_pointwise_error = max_abs_error
            best_epsilon = epsilon
            best_kernel_name = kernel_name
            best_W = W.copy()

    if best_epsilon is None or best_kernel_name is None:
        print("No suitable epsilon and kernel combination found.")
        return

    print(f"Best epsilon found: {best_epsilon:.5f}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Minimum max pointwise error: {min_max_pointwise_error:.5e}")

    # Final W with all data
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]

    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )
    Phi_train = kernel_func(dists_train, epsilon)
    Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

    U_, S_, Vt_ = np.linalg.svd(Phi_train, full_matrices=False)
    tolerance = 1e-8
    S_inv = np.diag([1/s if s > tolerance else 0 for s in S_])
    W = Vt_.T @ S_inv @ U_.T @ q_s_train

    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W,
            'q_p_train': q_p_train,
            'q_s_train': q_s_train,
            'epsilon': epsilon,
            'kernel_name': best_kernel_name
        }, f)
    print(f"Global weight matrix and data saved in {training_data_filename}.")

    print("Processing complete.")

if __name__ == '__main__':
    main()