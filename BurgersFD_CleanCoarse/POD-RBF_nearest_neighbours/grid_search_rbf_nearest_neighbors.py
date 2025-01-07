# grid_search_rbf_neighbors.py

import os
import numpy as np
import time
import pickle
import itertools
from sklearn.metrics import mean_squared_error
from scipy.spatial import KDTree
from sklearn.preprocessing import MinMaxScaler

# Import RBF kernels
def gaussian_rbf(r, epsilon):
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    return r

# RBF kernels dictionary
rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'linear': linear_rbf
}

# Interpolation logic (unchanged)
def interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new, epsilon, neighbors, kernel_func):
    dist, idx = kdtree.query(x_new, k=neighbors)
    X_neighbors = q_p_train[idx].reshape(neighbors, -1)
    Y_neighbors = q_s_train[idx].reshape(neighbors, -1)

    dists_neighbors = np.linalg.norm(X_neighbors[:, np.newaxis] - X_neighbors[np.newaxis, :], axis=-1)
    Phi_neighbors = kernel_func(dists_neighbors, epsilon)
    Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization
    #W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    # Perform SVD on Phi_neighbors
    U, S, Vt = np.linalg.svd(Phi_neighbors, full_matrices=False)

    # Regularization for singular values to avoid instability
    tolerance = 1e-4
    S_inv = np.diag([1/s if s > tolerance else 0 for s in S])

    # Compute W_neighbors using the pseudoinverse of Phi_neighbors
    W_neighbors = Vt.T @ S_inv @ U.T @ Y_neighbors

    rbf_values = kernel_func(dist, epsilon)
    return rbf_values @ W_neighbors

# Grid search function
def grid_search(kdtree, q_p_train, q_s_train, q_p_val, q_s_val, scaler, epsilon_values, neighbor_values, kernel_names):
    best_params = None
    lowest_error = float('inf')

    # Iterate through all combinations of parameters
    for epsilon, neighbors, kernel_name in itertools.product(epsilon_values, neighbor_values, kernel_names):
        kernel_func = rbf_kernels[kernel_name]

        # Compute validation error
        val_error = 0
        for x_new, y_true in zip(q_p_val, q_s_val):
            x_new_scaled = scaler.transform(x_new.reshape(1, -1))
            y_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new_scaled, epsilon, neighbors, kernel_func)

            # Reshape y_pred to match y_true if necessary
            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()

            val_error += mean_squared_error(y_true, y_pred)

        val_error /= len(q_p_val)  # Average error

        print(f"Epsilon: {epsilon}, Neighbors: {neighbors}, Kernel: {kernel_name}, Validation Error: {val_error:.5e}")

        # Track best parameters
        if val_error < lowest_error:
            lowest_error = val_error
            best_params = {'epsilon': epsilon, 'neighbors': neighbors, 'kernel_name': kernel_name}

    return best_params, lowest_error


# Main function
def main():
    # Load preprocessed data
    model_dir = "pod_rbf_nearest_model"
    with open(os.path.join(model_dir, 'training_data.pkl'), 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # Split data for validation
    train_size = int(0.8 * len(q_p_train))
    q_p_val, q_s_val = q_p_train[train_size:], q_s_train[train_size:]
    q_p_train, q_s_train = q_p_train[:train_size], q_s_train[:train_size]

    # Define grid search parameters
    epsilon_values = np.logspace(-3, 2, 20)  # Log-spaced epsilon values
    neighbor_values = np.unique(np.round(np.logspace(np.log10(5), np.log10(20), 10)).astype(int))  # Log-spaced neighbor values
    kernel_names = list(rbf_kernels.keys())  # Kernel options

    print(f"Epsilon values: {epsilon_values}")
    print(f"Neighbor values: {neighbor_values}")
    print(f"Kernel names: {kernel_names}")

    # Perform grid search
    best_params, lowest_error = grid_search(kdtree, q_p_train, q_s_train, q_p_val, q_s_val, scaler, epsilon_values, neighbor_values, kernel_names)

    print(f"Best Parameters: {best_params}")
    print(f"Lowest Validation Error: {lowest_error:.5e}")

    # Save the best parameters
    with open(os.path.join(model_dir, 'best_params.pkl'), 'wb') as f:
        pickle.dump(best_params, f)

    print("Best parameters saved successfully.")

if __name__ == "__main__":
    main()
