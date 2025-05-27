# compute_gp_models.py

import os
import numpy as np
import time
import pickle
import logging
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern  # Added Matern
)
from sklearn.multioutput import MultiOutputRegressor  # [Imported but not used now]
from scipy.optimize import fmin_l_bfgs_b

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
    total_snaps = snapshot_shape[1] * snap_count  # Total snapshots = time steps * param combos
    print(f"Total number of snapshots to aggregate: {total_snaps}")

    # Pre-allocate memory for all snapshots
    snaps = np.zeros((snapshot_shape[0], total_snaps))

    # Collect snapshots
    col_offset = 0
    successful_mu = []
    missing_mu = []

    for idx, mu in enumerate(mu_samples):
        try:
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
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
        print("Missing parameter sets have been logged in 'missing_snapshots.log'.")

    if snaps.size == 0:
        print("No snapshots were loaded. Exiting the workflow.")
        return

    print(f"Combined snapshot matrix shape: {snaps.shape}")

    # Compute or load POD basis
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
        print(f"Computed POD basis with method '{pod_method}' and {num_modes} modes.")

        np.save('basis.npy', basis)
        np.save('sigma.npy', sigma)
        print("Saved computed basis to 'basis.npy' and singular values to 'sigma.npy'.")

    # Select primary vs. secondary modes
    primary_modes = 10
    total_modes = 150

    # Project snapshots
    print("Projecting snapshots onto the POD basis...")
    projection_start_time = time.time()
    q = basis.T @ snaps
    q_p = q[:primary_modes, :]     # Primary mode projections
    q_s = q[primary_modes:total_modes, :]  # Secondary mode projections
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")

    # Scale q_p (input features)
    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T
    print("Normalization of q_p complete.")

    # Save the scaler for q_p
    model_dir = "pod_gp_model"  # Changed from "modes" to "pod_gp_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # -------------------------------------------------------------------------
    # COMMENT OUT THE OUTPUT SCALING
    # -------------------------------------------------------------------------
    # print("Scaling q_s data using StandardScaler...")
    # y_scaler = StandardScaler()
    # q_s_scaled = y_scaler.fit_transform(q_s.T).T
    # print("Scaling of q_s complete.")
    #
    # with open(os.path.join(model_dir, 'y_scaler.pkl'), 'wb') as f:
    #     pickle.dump(y_scaler, f)
    # print("y_scaler saved successfully.")
    #
    # We'll simply store q_s unscaled for training.
    # -------------------------------------------------------------------------

    # Save the unscaled q_s for future use
    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q.npy'), q)
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    # np.save(os.path.join(model_dir, 'q_s_scaled.npy'), q_s_scaled)  # commented out
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)  # Save unscaled q_s
    print("Primary and secondary modes, as well as projected data, saved successfully.")

    # Prepare training data
    X_train = q_p_normalized.T  # shape: (num_samples, num_primary_modes)
    # Y_train = q_s_scaled.T     # If we scaled q_s, but now we won't
    Y_train = q_s.T             # shape: (num_samples, num_secondary_modes) - no scaling

    # -------------------------------------------------------------------------
    # COMMENT THE RBF AND USE A MATÉRN KERNEL INSTEAD
    # -------------------------------------------------------------------------
    #kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-6, 1e-3))
    #
    #kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3,1e2)) * \
    #         Matern(length_scale=0.5, length_scale_bounds=(1e-2, 5.0), nu=1.5) 
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")* \
             Matern(0.5 * np.ones(primary_modes), (1e-2, 5.0), nu=1.5) 
    # -------------------------------------------------------------------------

    # Instantiate the base GP regressor
    base_gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-8,
        n_restarts_optimizer=1,
        optimizer='fmin_l_bfgs_b',
        normalize_y=False
    )

    # Single GPR handling multi-dimensional outputs
    print("Training single GPR with multi-dimensional outputs (Matérn kernel, unscaled q_s)...")
    start_time = time.time()
    base_gp.fit(X_train, Y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Save the single multi-output GP
    gp_models_filename = os.path.join(model_dir, 'gp_model.pkl')
    with open(gp_models_filename, 'wb') as f:
        pickle.dump(base_gp, f)
    print(f"GP model saved successfully in {gp_models_filename}.")

    print("Processing complete.")

    print(base_gp.kernel_.get_params())

if __name__ == '__main__':
    main()
