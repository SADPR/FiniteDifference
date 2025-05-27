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
    primary_modes = 20
    total_modes = 150  # Ensure total_modes >= primary_modes

    # Project the snapshots onto the POD basis
    print("Projecting snapshots onto the POD basis...")
    projection_start_time = time.time()
    q = basis.T @ snaps  # Project snapshots onto the POD basis
    q_p = q[:primary_modes, :]  # Primary mode projections
    q_s = q[primary_modes:total_modes, :]  # Secondary mode projections
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")
    del snaps

    np.save('q_p.npy', q_p)
    np.save('q_s.npy', q_s)  # Note: This line seems to overwrite q_p; consider using a different name for q_s if needed.
    print("Saving complete.")

    # -------------------------------------------------------------------
    # Added section: Project test snapshots for the new parameter sets.
    # Test parameters: [mu1, mu2] pairs: [4.75, 0.02], [4.56, 0.019], [5.19, 0.026]
    test_params = [
        [4.75, 0.02],
        [4.56, 0.019],
        [5.19, 0.026]
    ]
    print("Projecting test snapshots onto the POD basis...")

    # Pre-allocate test snapshots matrix (each test snapshot has the same shape as training snapshots)
    n_test = len(test_params)
    total_test_snaps = snapshot_shape[1] * n_test
    snaps_test = np.zeros((snapshot_shape[0], total_test_snaps))
    col_offset = 0
    for mu in test_params:
        try:
            test_snap = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=snap_folder)
            snaps_test[:, col_offset:col_offset + test_snap.shape[1]] = test_snap
            col_offset += test_snap.shape[1]
            print(f"Loaded test snapshot for mu1={mu[0]}, mu2={mu[1]}")
        except FileNotFoundError as e:
            print(f"Test snapshot for mu1={mu[0]}, mu2={mu[1]} not found: {e}")
            logging.info(f"Missing test snapshot for mu1={mu[0]}, mu2={mu[1]}")

    # Project the aggregated test snapshots onto the POD basis
    q_test = basis.T @ snaps_test
    # Split test projection into primary and secondary modes
    q_p_test = q_test[:primary_modes, :]
    q_s_test = q_test[primary_modes:total_modes, :]

    # Save test projections in the current directory
    np.save('q_p_test.npy', q_p_test)
    np.save('q_s_test.npy', q_s_test)
    print(f"Test projections saved with shapes: q_p_test {q_p_test.shape}, q_s_test {q_s_test.shape}")
    # -------------------------------------------------------------------

if __name__ == '__main__':
    main()


