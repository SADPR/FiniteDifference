import os
import numpy as np
import time
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold
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

def matern_kernel(r, epsilon):
    """Matérn kernel function with nu=3/2.
    
    k(r) = (1 + sqrt(3)*epsilon*r) * exp(-sqrt(3)*epsilon*r)
    """
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * epsilon * r) * np.exp(-sqrt3 * epsilon * r)

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'matern': matern_kernel
    
}

def main():

    # Load a precomputed basis
    basis_path = 'basis.npy'
    basis = np.load(basis_path, allow_pickle=True)
    print(f"Loaded precomputed basis from {basis_path}.")
    
    # Define how many primary modes to use
    primary_modes = 10
    total_modes = 150  # Ensure total_modes >= primary_modes

    # -------------------------------------------------------------------
    # Added section: Project test snapshots for the new parameter sets.
    # Test parameters: [mu1, mu2] pairs: [4.75, 0.02], [4.56, 0.019], [5.19, 0.026]
    test_params = [
        [4.56, 0.019]
    ]
    print("Projecting test snapshots onto the POD basis...")

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Pre-allocate test snapshots matrix (each test snapshot has the same shape as training snapshots)
    n_test = len(test_params)
    total_test_snaps = 501 * n_test
    snaps_test = np.zeros((basis.shape[0], total_test_snaps))
    col_offset = 0
    for mu in test_params:
        test_snap = load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)
        snaps_test[:, col_offset:col_offset + test_snap.shape[1]] = test_snap
        col_offset += test_snap.shape[1]
        print(f"Loaded test snapshot for mu1={mu[0]}, mu2={mu[1]}")

    # Project the aggregated test snapshots onto the POD basis
    q_test = basis.T @ snaps_test
    q_p_test_original = q_test[:primary_modes, :]
    q_s_test = q_test[primary_modes:total_modes, :]

    q_p = np.load('q_p.npy')
    q_s = np.load('q_s.npy')
    print('Shape of q_p: ',q_p.shape)
    print('Shape of q_s: ',q_s.shape)
    print('Shape of q_p_test: ',q_p_test_original.shape)
    print('Shape of q_s_test: ',q_s_test.shape)

    q_p = np.hstack((q_p, q_p_test_original))
    q_s = np.hstack((q_s, q_s_test))

    print('Shape of q_p: ',q_p.shape)
    print('Shape of q_s: ',q_s.shape)
    
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
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q_p_normalized, q_s), saved successfully.")

    # Prepare training data (each row is a snapshot)
    q_p_train = q_p_normalized.T  # Shape: (n_snapshots, num_primary_modes)
    q_s_train = q_s.T             # Shape: (n_snapshots, num_secondary_modes)

    # Split test projection into primary and secondary modes
    q_p_test_original = q_test[:primary_modes, :]
    q_s_test = q_test[primary_modes:total_modes, :]
    q_p_test = scaler.transform(q_p_test_original.T).T
    q_p_test = q_p_test[:,:]
    q_s_test = q_s_test[:,:]
    q_p_test = q_p_test.T  # Now shape: (n_test_snapshots, num_primary_modes)
    q_s_test = q_s_test.T  # Now shape: (n_test_snapshots, num_secondary_modes)

    # Define the search space for Bayesian optimization
    space = [
        Real(np.log(1e-2), np.log(10), name='log_epsilon'),
        Categorical(list(rbf_kernels.keys()), name='kernel_name'),
        Real(np.log(1e-10), np.log(1e-3), name='log_lambda_val')
    ]

    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )

    # Compute distance matrix for test data relative to training data
    dists_test = np.linalg.norm(
        q_p_test[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )

    print("Optimizing epsilon, kernel and lambda using a train–validation split and bayesian optimization...")
    # Define the objective function
    @use_named_args(space)
    def objective(log_epsilon, kernel_name, log_lambda_val):
        epsilon = np.exp(log_epsilon)  # Optimize in log-space to ensure positivity
        kernel_func = rbf_kernels[kernel_name]
        lambda_val = np.exp(log_lambda_val)
            
        kernel_func = rbf_kernels[kernel_name]
        errors = []
        rpes = []

        Phi_train = kernel_func(dists_train, epsilon)
        Phi_train += np.eye(Phi_train.shape[0]) * lambda_val  # Regularization

        try:
            W_cv = np.linalg.solve(Phi_train, q_s_train)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}, lambda={lambda_val}. Skipping this fold.")
            return np.inf

        Phi_val = kernel_func(dists_test, epsilon)
        y_val_pred = Phi_val @ W_cv

        error = mean_squared_error(q_s_test, y_val_pred)
        rpe = (np.linalg.norm(q_s_test - y_val_pred, 'fro') /
                    np.linalg.norm(q_s_test, 'fro')) * 100

        print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, Lambda: {lambda_val}, "
                f"CV MSE: {error:.5e}, CV Relative Error: {rpe:.2f}%")

        return rpe

    # Start timing
    start_time = time.time()

    # Perform Bayesian optimization
    print("Optimizing epsilon and kernel using Bayesian optimization...")
    result = gp_minimize(
        objective,
        space,
        acq_func='EI',  # Expected Improvement
        n_calls=100,
        random_state=42,
        verbose=True,
        n_random_starts = 9
    )

    # End timing
    end_time = time.time()

    # Extract results
    best_log_epsilon = result.x[0]
    best_epsilon = np.exp(best_log_epsilon)
    best_kernel_name = result.x[1]
    best_log_labmda_val = result.x[2]
    best_lambda_val = np.exp(best_log_labmda_val)

    # Display results
    print(f"Best epsilon found: {best_epsilon:.5e}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")

    # Use the best hyperparameters to compute final W using all training data (q_p_train and q_s_train)
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]
    lambda_val = best_lambda_val

    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )
    Phi_train = kernel_func(dists_train, epsilon)
    Phi_train += np.eye(Phi_train.shape[0]) * lambda_val

    try:
        W = np.linalg.solve(Phi_train, q_s_train)
    except np.linalg.LinAlgError:
        print("LinAlgError when solving for the final weight matrix. Exiting.")
        return

    # Save the global weight matrix and related data
    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W,                        # Global weight matrix
            'q_p_train': q_p_train,        # Normalized primary training coordinates
            'q_s_train': q_s_train,        # Secondary training outputs
            'epsilon': epsilon,            # Best epsilon
            'kernel_name': best_kernel_name,  # Best kernel name
            'lambda': lambda_val           # Best lambda
        }, f)
    print(f"Global weight matrix and data saved in {training_data_filename}.")

    print("Processing complete.")

if __name__ == '__main__':
    main()
