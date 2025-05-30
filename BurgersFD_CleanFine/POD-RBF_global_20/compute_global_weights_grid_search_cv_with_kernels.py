
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

#############################
# Selected RBF Kernels
#############################

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'multiquadric': multiquadric_rbf,
    'linear': linear_rbf
}

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

    # Load a precomputed basis
    basis_path = 'basis.npy'
    if os.path.exists(basis_path):
        basis = np.load(basis_path, allow_pickle=True)
        print(f"Loaded precomputed basis from {basis_path}.")
    else:
        print(f"Basis file '{basis_path}' not found. Please compute the basis first.")
        return

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

    # Normalize q_p using Min-Max normalization and save the scaler
    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T  # Note the transpose operations
    print("Normalization complete.")

    # Save the scaler for future use
    model_dir = "pod_rbf_global_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Save the normalized q_p and q_s for future use
    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q.npy'), q)
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q, q_p_normalized, q_s), saved successfully.")

    # Prepare training data
    q_p_train = q_p_normalized.T  # Shape: (num_snapshots, num_primary_modes)
    q_s_train = q_s.T  # Shape: (num_snapshots, num_secondary_modes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.1, random_state=42
    )

    # Define grids for epsilon and kernel names
    epsilon_values = np.logspace(np.log10(0.01), np.log10(10), 100)
    kernel_names = list(rbf_kernels.keys())

    best_epsilon = None
    best_kernel_name = None
    lowest_error = np.inf
    best_W = None

    start_time = time.time()
    print("Optimizing epsilon and kernel using grid search...")
    for epsilon, kernel_name in product(epsilon_values, kernel_names):
        kernel_func = rbf_kernels[kernel_name]

        # Compute distance matrix between training points
        dists_train = np.linalg.norm(
            X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        # Compute kernel matrix Phi_train
        Phi_train = kernel_func(dists_train, epsilon)
        # Add regularization
        Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

        # Solve for W
        try:
            W = np.linalg.solve(Phi_train, y_train)
            #print(f'Condition number is: {np.linalg.cond(Phi_train)}')
            #U, S, Vt = np.linalg.svd(Phi_train, full_matrices=False)
            #tolerance = 1e-6  # Regularization threshold for singular values
            #S_inv = np.diag([1/s if s > tolerance else 0 for s in S])
            #W = Vt.T @ S_inv @ U.T @ y_train
            #print(f'The norm of W is: {np.linalg.norm(W)}')
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}. Skipping.")
            continue

        # Compute distance matrix between validation and training points
        dists_val = np.linalg.norm(
            X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        # Compute kernel matrix Phi_val
        Phi_val = kernel_func(dists_val, epsilon)

        # Predict on validation set
        y_val_pred = Phi_val @ W

        # Compute validation error
        error = mean_squared_error(y_val, y_val_pred)
        #error = np.linalg.norm(y_val - y_val_pred) / np.linalg.norm(y_val)
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
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    # Use the best epsilon and kernel to compute final W using all data
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]

    # Compute distance matrix between all training points
    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )
    # Compute kernel matrix Phi_train
    Phi_train = kernel_func(dists_train, epsilon)
    # Add regularization
    Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

    # Solve for W
    #U, S, Vt = np.linalg.svd(Phi_train, full_matrices=False)
    #tolerance = 1e-6  # Regularization threshold for singular values
    #S_inv = np.diag([1/s if s > tolerance else 0 for s in S])
    #W = Vt.T @ S_inv @ U.T @ q_s_train
    W = np.linalg.solve(Phi_train, q_s_train)

    # Save the global weight matrix and necessary data
    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W,                       # Global weight matrix
            'q_p_train': q_p_train,       # Normalized primary training coordinates
            'q_s_train': q_s_train,       # Secondary training outputs
            'epsilon': epsilon,           # Best epsilon
            'kernel_name': best_kernel_name  # Best kernel name
        }, f)
    print(f"Global weight matrix and data saved in {training_data_filename}.")

    print("Processing complete.")


if __name__ == '__main__':
    main()

