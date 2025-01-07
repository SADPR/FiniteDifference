import os
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import KDTree
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

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
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    rbf_values = kernel_func(dist, epsilon)
    return rbf_values @ W_neighbors

# Objective function for Bayesian Optimization
def objective_function(params, kdtree, q_p_train, q_s_train, q_p_val, q_s_val, scaler):
    log_epsilon, neighbors, kernel_name = params
    epsilon = np.exp(log_epsilon)  # Convert from log-space
    kernel_func = rbf_kernels[kernel_name]

    val_error = 0
    for x_new, y_true in zip(q_p_val, q_s_val):
        x_new_scaled = scaler.transform(x_new.reshape(1, -1))
        y_pred = interpolate_on_the_fly(kdtree, q_p_train, q_s_train, x_new_scaled, epsilon, neighbors, kernel_func)

        # Reshape y_pred to match y_true if necessary
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()

        val_error += mean_squared_error(y_true, y_pred)

    val_error /= len(q_p_val)  # Average error
    print(f"Epsilon: {epsilon:.5e}, Neighbors: {neighbors}, Kernel: {kernel_name}, Validation Error: {val_error:.5e}")

    return val_error

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

    # Define search space for Bayesian Optimization
    space = [
        Real(np.log(1e-3), np.log(1e1), name='log_epsilon'),  # Epsilon in log-space
        Integer(5, 100, name='neighbors'),                    # Number of neighbors
        Categorical(list(rbf_kernels.keys()), name='kernel_name')  # Kernel name
    ]

    # Define Bayesian Optimization function
    @use_named_args(space)
    def objective(**params):
        return objective_function(
            [params['log_epsilon'], params['neighbors'], params['kernel_name']],
            kdtree, q_p_train, q_s_train, q_p_val, q_s_val, scaler
        )

    # Perform Bayesian Optimization
    print("Starting Bayesian Optimization...")
    result = gp_minimize(
        objective,
        space,
        acq_func='EI',  # Expected Improvement
        n_calls=300,
        random_state=42,
        verbose=True
    )

    # Extract best parameters
    best_params = {
        'epsilon': np.exp(result.x[0]),
        'neighbors': result.x[1],
        'kernel_name': result.x[2]
    }
    lowest_error = result.fun

    print(f"Best Parameters: {best_params}")
    print(f"Lowest Validation Error: {lowest_error:.5e}")

    # Save the best parameters
    with open(os.path.join(model_dir, 'best_params.pkl'), 'wb') as f:
        pickle.dump(best_params, f)

    print("Best parameters saved successfully.")

if __name__ == "__main__":
    main()
