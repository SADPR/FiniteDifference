import os
import numpy as np
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from itertools import product

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
    'imq': inverse_multiquadric_rbf
    #'multiquadric': multiquadric_rbf,
    #'linear': linear_rbf,
    #'matern': matern_kernel
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T  # q_p_normalized shape: (num_features, n_snapshots)
    print("Normalization complete.")

    # --- New: Remove near-duplicate snapshots from both q_p_normalized and q_s ---
    # Since snapshots are stored as columns, transpose first so that each row is a snapshot.
    q_p_normalized_T = q_p_normalized.T  # Shape: (n_snapshots, num_features)
    q_s_T = q_s.T                        # Shape: (n_snapshots, num_secondary_modes)

    def get_duplicate_mask(q_data, delta):
        """
        Returns a boolean mask for q_data where True indicates snapshots to keep.
        
        q_data: np.ndarray of shape (n_snapshots, d)
        delta: float, threshold for similarity.
        """
        n_samples = q_data.shape[0]
        keep = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            if not keep[i]:
                continue
            for j in range(i + 1, n_samples):
                if keep[j] and np.linalg.norm(q_data[i] - q_data[j]) < delta:
                    keep[j] = False
        return keep

    delta = 0.01  # Adjust this threshold as needed
    print(q_p_normalized.shape)
    mask = get_duplicate_mask(q_p_normalized_T, delta)
    print(f"Removing {q_p_normalized_T.shape[0] - np.sum(mask)} near-duplicate snapshots.")

    # Apply mask and transpose back to recover the original shape
    q_p_normalized = q_p_normalized_T[mask].T  # Now q_p_normalized has shape (num_features, n_clean_snapshots)
    q_s = q_s_T[mask].T                        # Similarly, q_s now contains only the corresponding snapshots
    print(q_p_normalized.shape)

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
    # Training data is expected to have each row as a snapshot.
    q_p_train = q_p_normalized.T  # Shape: (n_train_snapshots, num_primary_modes)
    q_s_train = q_s.T             # Shape: (n_train_snapshots, num_secondary_modes)

    # Load test projections and adjust orientation (so each row is a snapshot)
    q_p_test = np.load('q_p_test.npy')
    q_s_test = np.load('q_s_test.npy')
    q_p_test = scaler.transform(q_p_test.T).T
    q_p_test = q_p_test.T  # Now shape: (n_test_snapshots, num_primary_modes)
    q_s_test = q_s_test.T  # Now shape: (n_test_snapshots, num_secondary_modes)

    # Define grids for epsilon, lambda and kernel names
    epsilon_values = np.logspace(np.log10(1), np.log10(10.0), 40)
    lambda_values = [1e-8, 1e-6]
    kernel_names = list(rbf_kernels.keys())

    best_epsilon = None
    best_kernel_name = None
    best_lambda = None
    lowest_error = np.inf
    best_rel_error = None
    best_W = None

    start_time = time.time()
    print("Optimizing epsilon, kernel and lambda using test data for validation...")
    # Compute distance matrix on training data
    dists_train = np.linalg.norm(
        q_p_train[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )

    # Compute distance matrix for test data relative to training data
    dists_test = np.linalg.norm(
        q_p_test[:, np.newaxis, :] - q_p_train[np.newaxis, :, :], axis=2
    )
    for epsilon, kernel_name, lambda_val in product(epsilon_values, kernel_names, lambda_values):
        kernel_func = rbf_kernels[kernel_name]
        
        Phi_train = kernel_func(dists_train, epsilon)
        # Add regularization using lambda_val
        Phi_train += np.eye(Phi_train.shape[0]) * lambda_val

        try:
            W = np.linalg.solve(Phi_train, q_s_train)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at epsilon={epsilon:.5f}, kernel={kernel_name}, lambda={lambda_val}. Skipping.")
            continue

        Phi_test = kernel_func(dists_test, epsilon)
        y_test_pred = Phi_test @ W

        # Compute error metrics on test data
        test_error = mean_squared_error(q_s_test, y_test_pred)
        test_rel_error = (np.linalg.norm(q_s_test - y_test_pred, 'fro') /
                          np.linalg.norm(q_s_test, 'fro')) * 100
        print(f"Epsilon: {epsilon:.5f}, Kernel: {kernel_name}, Lambda: {lambda_val}, "
              f"Test MSE: {test_error:.5e}, Test Relative Error: {test_rel_error:.2f}%")

        if test_error < lowest_error:
            lowest_error = test_error
            best_epsilon = epsilon
            best_kernel_name = kernel_name
            best_lambda = lambda_val
            best_rel_error = test_rel_error
            best_W = W.copy()

    if best_epsilon is None or best_kernel_name is None:
        print("No suitable hyperparameter combination found. Exiting.")
        return

    print(f"\nBest epsilon found: {best_epsilon:.5f}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Best lambda found: {best_lambda}")
    print(f"Lowest Test MSE: {lowest_error:.5e}")
    print(f"Lowest Test Relative Error: {best_rel_error:.2f}%")
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
