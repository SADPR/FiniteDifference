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

def matern_kernel(r, epsilon):
    """Matérn kernel function with nu=3/2.
    
    k(r) = (1 + sqrt(3)*epsilon*r) * exp(-sqrt(3)*epsilon*r)
    """
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * epsilon * r) * np.exp(-sqrt3 * epsilon * r)

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    #'multiquadric': multiquadric_rbf,
    #'linear': linear_rbf,
    'matern': matern_kernel
}

def main():
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

    q_p = np.load('q_p.npy')
    q_s = np.load('q_s.npy')
    
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

    # Create a train–validation split (90% training, 10% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        q_p_train, q_s_train, test_size=0.2, random_state=42
    )

    # Define grids for epsilon, lambda and kernel names
    epsilon_values = np.logspace(np.log10(0.1), np.log10(10), 20)
    lambda_values = [1e-10, 1e-8]
    kernel_names = list(rbf_kernels.keys())

    best_epsilon = None
    best_kernel_name = None
    best_lambda = None
    lowest_error = np.inf
    best_rel_error = None
    best_W = None

    start_time = time.time()
    # Compute distance matrix for the training split
    dists_train = np.linalg.norm(
        X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
    )
    print("Optimizing epsilon, kernel and lambda using a train–validation split...")
    for epsilon, kernel_name, lambda_val in product(epsilon_values, kernel_names, lambda_values):
        kernel_func = rbf_kernels[kernel_name]
        
        Phi_train = kernel_func(dists_train, epsilon)
        Phi_train += np.eye(Phi_train.shape[0]) * lambda_val  # Regularization

        try:
            W_cv = np.linalg.solve(Phi_train, y_train)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}, lambda={lambda_val}. Skipping.")
            continue

        # Compute distance matrix for the validation split (relative to training)
        dists_val = np.linalg.norm(
            X_val[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2
        )
        Phi_val = kernel_func(dists_val, epsilon)
        y_val_pred = Phi_val @ W_cv

        # Compute validation errors
        fold_error = mean_squared_error(y_val, y_val_pred)
        fold_rpe = (np.linalg.norm(y_val - y_val_pred, 'fro') /
                    np.linalg.norm(y_val, 'fro')) * 100

        print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, Lambda: {lambda_val}, "
              f"Val MSE: {fold_error:.5e}, Val Relative Error: {fold_rpe:.2f}%")

        if fold_error < lowest_error:
            lowest_error = fold_error
            best_epsilon = epsilon
            best_kernel_name = kernel_name
            best_lambda = lambda_val
            best_rel_error = fold_rpe
            best_W = W_cv.copy()

    if best_epsilon is None or best_kernel_name is None:
        print("No suitable hyperparameter combination found. Exiting.")
        return

    print(f"\nBest epsilon found: {best_epsilon:.5f}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Best lambda found: {best_lambda}")
    print(f"Lowest Val MSE: {lowest_error:.5e}")
    print(f"Lowest Val Relative Error: {best_rel_error:.2f}%")
    print(f"Hyperparameter optimization completed in {time.time() - start_time:.2f} seconds.")

    # Use the best hyperparameters to compute final W using all training data (q_p_train and q_s_train)
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]
    lambda_val = best_lambda

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
