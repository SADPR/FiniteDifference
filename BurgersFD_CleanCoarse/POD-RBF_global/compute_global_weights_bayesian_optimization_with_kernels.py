# compute_global_weights_bayesian_optimization_with_kernels.py

import os
import numpy as np
import time
import pickle
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps, make_2D_grid

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

# Dictionary mapping kernel names to functions
rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'multiquadric': multiquadric_rbf
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
        return  # Exit since no snapshots are available

    # Generate all parameter samples
    mu_samples = get_snapshot_params()
    print(f"Total parameter samples: {len(mu_samples)}")

    # Attempt to load the shape of the first snapshot to determine snapshot dimensions
    try:
        first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
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
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
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

    # Normalize q_p using Min-Max normalization and save the scaler
    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T  # Note the transpose operations
    print("Normalization complete.")

    # Save the scaler for future use
    modes_dir = "modes"
    if not os.path.exists(modes_dir):
        os.makedirs(modes_dir)
        print(f"Created modes directory: {modes_dir}")

    with open(os.path.join(modes_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Save the normalized q_p and q_s for future use
    np.save(os.path.join(modes_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(modes_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(modes_dir, 'q.npy'), q)
    np.save(os.path.join(modes_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(modes_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q, q_p_normalized, q_s), saved successfully.")

    # Prepare training data
    q_p_train = q_p_normalized.T  # Shape: (num_snapshots, num_primary_modes)
    q_s_train = q_s.T  # Shape: (num_snapshots, num_secondary_modes)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.2, random_state=42
    )

    # Define the search space for Bayesian optimization
    space = [
        Real(np.log(1e-3), np.log(1e1), name='log_epsilon'),
        Categorical(list(rbf_kernels.keys()), name='kernel_name')
    ]

    # Define the objective function
    @use_named_args(space)
    def objective(log_epsilon, kernel_name):
        epsilon = np.exp(log_epsilon)  # Optimize in log-space to ensure positivity
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
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5e}, kernel={kernel_name}. Returning infinity.")
            return np.inf  # Return infinity if matrix is singular

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
        print(f"Epsilon: {epsilon:.5e}, Kernel: {kernel_name}, Validation MSE: {error:.5e}")
        return error

    # Perform Bayesian optimization
    print("Optimizing epsilon and kernel using Bayesian optimization...")
    result = gp_minimize(
        objective,
        space,
        acq_func='EI',  # Expected Improvement
        n_calls=30,
        random_state=42,
        verbose=True
    )
    best_log_epsilon = result.x[0]
    best_epsilon = np.exp(best_log_epsilon)
    best_kernel_name = result.x[1]
    print(f"Best epsilon found: {best_epsilon:.5e}")
    print(f"Best kernel found: {best_kernel_name}")

    # Compute final W using the best epsilon and kernel
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]

    # Compute distance matrix between training points
    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )
    # Compute kernel matrix Phi_train
    Phi_train = kernel_func(dists_train, epsilon)
    # Add regularization
    Phi_train += np.eye(Phi_train.shape[0]) * 1e-8

    # Solve for W
    W = np.linalg.solve(Phi_train, q_s_train)

    # Save the global weight matrix and necessary data
    training_data_filename = os.path.join(modes_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W,
            'q_p_train': q_p_train,
            'epsilon': epsilon,
            'kernel_name': best_kernel_name
        }, f)
    print(f"Global weight matrix and data saved in {training_data_filename}.")

    print("Processing complete.")

if __name__ == '__main__':
    main()
