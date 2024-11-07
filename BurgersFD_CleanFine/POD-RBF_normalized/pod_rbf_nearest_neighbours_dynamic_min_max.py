import os
import numpy as np
import time
import pickle
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import logging
import sys

# Determine the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path if it's not already present
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now, import the required functions from hypernet2D
from hypernet2D import load_or_compute_snaps, make_2D_grid

def get_snapshot_params():
    """
    Generate a list of parameter vectors [mu1, mu2] within specified ranges.

    Returns:
    - mu_samples: List of [mu1, mu2] pairs.
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

def param_to_snap_fn(mu, snap_folder="param_snaps", extension=".npy"):
    """
    Generates a filename for snapshots based on the parameter vector mu.

    Parameters:
    - mu: Parameter vector [mu1, mu2].
    - snap_folder: Directory where snapshots are stored.
    - extension: File extension for snapshot files.

    Returns:
    - snap_fn: Filename string for the given parameters.
    """
    # Ensure mu has two elements
    assert len(mu) == 2, "Parameter vector mu must have two elements [mu1, mu2]."

    # Format the filename with mu values, replacing decimal points with 'p' for filesystem compatibility
    mu1_str = f"{mu[0]:.6f}".replace('.', 'p')
    mu2_str = f"{mu[1]:.6f}".replace('.', 'p')
    snap_fn = os.path.join(snap_folder, f"snap_mu1_{mu1_str}_mu2_{mu2_str}{extension}")
    return snap_fn

def perform_pod(snaps, num_modes=150, method='rsvd', random_state=None):
    """
    Perform Proper Orthogonal Decomposition (POD) using SVD or Randomized SVD.

    Parameters:
    - snaps: Snapshot matrix of shape (total_dofs, total_snapshots).
    - num_modes: Total number of POD modes to retain.
    - method: 'svd' for standard SVD or 'rsvd' for randomized SVD.
    - random_state: Random state for reproducibility (only used if method='rsvd').

    Returns:
    - basis: POD basis matrix of shape (total_dofs, num_modes).
    - sigma: Singular values array of shape (num_modes,).
    """
    if method == 'rsvd':
        print("Performing Randomized SVD for POD...")
        start_time = time.time()
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

def remove_duplicates(data, tolerance=1e-8):
    """Remove near-duplicate rows from data."""
    _, unique_indices = np.unique(np.round(data / tolerance) * tolerance, axis=0, return_index=True)
    return unique_indices

def build_kdtree_and_save_data(q_p_normalized, q_s, filename):
    """
    Build a KDTree using unique normalized q_p points and save it along with q_p and q_s.

    Parameters:
    - q_p_normalized: Normalized projected primary mode coordinates (shape: primary_modes x total_snapshots).
    - q_s: Projected secondary mode coordinates (shape: secondary_modes x total_snapshots).
    - filename: Filename for saving the KDTree and training data.
    """
    # Step 1: Remove duplicates
    unique_indices = remove_duplicates(q_p_normalized.T)
    q_p_unique = q_p_normalized[:, unique_indices]
    q_s_unique = q_s[:, unique_indices]

    # Step 2: Build the KDTree using unique normalized q_p points
    print("Building KDTree...")
    kdtree_start_time = time.time()
    kdtree = KDTree(q_p_unique.T)
    print(f"KDTree construction took {time.time() - kdtree_start_time:.2f} seconds.")

    # Step 3: Save the KDTree and q_p, q_s data for future interpolation
    with open(filename, 'wb') as f:
        pickle.dump({'KDTree': kdtree, 'q_p': q_p_unique.T, 'q_s': q_s_unique.T}, f)

    print(f"Training data and KDTree have been saved successfully in {filename}.")

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

    # Compute the basis by collecting snapshots over a range of parameters
    snap_count = len(mu_samples)  # Total number of parameter combinations

    # Attempt to load the shape of the first snapshot to determine snapshot dimensions
    try:
        first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        snapshot_shape = first_snap.shape
        print(f"Shape of each snapshot: {snapshot_shape}")
    except FileNotFoundError as e:
        print(f"Error loading the first snapshot: {e}")
        print("Ensure that at least one snapshot exists to determine the snapshot dimensions.")
        # Log the error
        logging.error(f"Error loading the first snapshot: {e}")
        return  # Exit since snapshot dimensions are unknown

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
            print(snap_mu.shape)
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
    compute_basis = True  # Set to False to load a precomputed basis

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
        pod_method = 'rsvd'  # Choose between 'svd' (standard SVD) or 'rsvd' (randomized SVD)
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

    # Build KDTree and save training data using normalized q_p
    training_data_filename = os.path.join(modes_dir, 'training_data.pkl')
    build_kdtree_and_save_data(q_p_normalized, q_s, training_data_filename)

    print("Processing complete.")

if __name__ == '__main__':
    main()
