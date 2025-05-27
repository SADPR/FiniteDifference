import os
import numpy as np
import time
import pickle
import logging
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
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

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf
    #'multiquadric': multiquadric_rbf
    #'linear': linear_rbf,
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

    # Prepare training data
    q_p_train = q_p_normalized.T  # Shape: (num_snapshots, num_primary_modes)
    q_s_train = q_s.T             # Shape: (num_snapshots, num_secondary_modes)

    # Define grids for epsilon, lambda and kernel names
    epsilon_values = np.logspace(np.log10(0.1), np.log10(10), 20)
    lambda_values = [1e-10, 1e-8]
    kernel_names = list(rbf_kernels.keys())

    # Set up 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_epsilon = None
    best_kernel_name = None
    best_lambda = None
    lowest_error = np.inf
    best_rel_error = None
    best_W = None

    start_time = time.time()
    print("Optimizing epsilon, kernel and lambda using 5-fold cross validation...")
    for epsilon, kernel_name, lambda_val in product(epsilon_values, kernel_names, lambda_values):
        kernel_func = rbf_kernels[kernel_name]
        cv_errors = []
        cv_rel_errors = []
        # Perform 5-fold cross validation
        for train_index, val_index in kf.split(q_p_train):
            X_train_cv, X_val_cv = q_p_train[train_index], q_p_train[val_index]
            y_train_cv, y_val_cv = q_s_train[train_index], q_s_train[val_index]
            
            # Compute distance matrix for training fold
            dists_train = np.linalg.norm(
                X_train_cv[:, np.newaxis, :] - X_train_cv[np.newaxis, :, :], axis=2
            )
            Phi_train = kernel_func(dists_train, epsilon)
            # Add regularization using lambda_val
            Phi_train += np.eye(Phi_train.shape[0]) * lambda_val

            # Solve for local weight matrix W_cv
            try:
                W_cv = np.linalg.solve(Phi_train, y_train_cv)
            except np.linalg.LinAlgError:
                print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}, lambda={lambda_val}. Skipping fold.")
                continue

            # Compute distance matrix for validation fold
            dists_val = np.linalg.norm(
                X_val_cv[:, np.newaxis, :] - X_train_cv[np.newaxis, :, :], axis=2
            )
            Phi_val = kernel_func(dists_val, epsilon)
            y_val_pred = Phi_val @ W_cv

            # Compute and store validation error for this fold (MSE)
            fold_error = mean_squared_error(y_val_cv, y_val_pred)
            cv_errors.append(fold_error)
            
            # Compute and store relative percentage error for this fold
            fold_rpe = (np.linalg.norm(y_val_cv - y_val_pred, 'fro') / 
                        np.linalg.norm(y_val_cv, 'fro')) * 100
            cv_rel_errors.append(fold_rpe)
        
        # Average errors over the folds
        if cv_errors and cv_rel_errors:
            avg_error = np.mean(cv_errors)
            avg_rel_error = np.mean(cv_rel_errors)
            print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, Lambda: {lambda_val}, "
                  f"CV MSE: {avg_error:.5e}, CV Relative Error: {avg_rel_error:.2f}%")

            if avg_error < lowest_error:
                lowest_error = avg_error
                best_epsilon = epsilon
                best_kernel_name = kernel_name
                best_lambda = lambda_val
                best_rel_error = avg_rel_error

    if best_epsilon is None or best_kernel_name is None:
        print("No suitable hyperparameter combination found. Exiting.")
        return

    print(f"\nBest epsilon found: {best_epsilon:.5f}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Best lambda found: {best_lambda}")
    print(f"Lowest CV MSE: {lowest_error:.5e}")
    print(f"Lowest CV Relative Error: {best_rel_error:.2f}%")
    print(f"Hyperparameter optimization completed in {time.time() - start_time:.2f} seconds.")

    # Use the best epsilon, kernel and lambda to compute final W using all training data
    epsilon = best_epsilon
    kernel_func = rbf_kernels[best_kernel_name]
    lambda_val = best_lambda

    # Compute distance matrix between all training points
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

    # Save the global weight matrix and necessary data
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
