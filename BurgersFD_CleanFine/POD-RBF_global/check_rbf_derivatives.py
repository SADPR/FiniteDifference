# check_rbf_derivatives_global.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

# Define the RBF kernels
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def imq_rbf(r, epsilon):
    """Inverse Multiquadric (IMQ) RBF kernel function."""
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def mq_rbf(r, epsilon):
    """Multiquadric (MQ) RBF kernel function."""
    return np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    """Linear RBF kernel function."""
    return r

def interpolate_and_compute_jacobian(
    kernel_type, rbf_func, q_p_train_norm, W_neighbors,
    q_p_sample, epsilon, scaler
):
    """
    Interpolate at a new point and compute the Jacobian using the global approach.

    Parameters:
    - kernel_type: str, type of RBF kernel ('gaussian', 'imq', 'mq')
    - rbf_func: function, RBF function corresponding to the kernel_type
    - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim)
    - W_neighbors: np.ndarray, precomputed weights (num_train x output_dim)
    - q_p_sample: np.ndarray, single sample primary coordinates (1 x dim)
    - epsilon: float, RBF parameter
    - scaler: MinMaxScaler object used for normalization

    Returns:
    - f_new: np.ndarray, interpolated output for the sample (output_dim,)
    - jacobian: np.ndarray, Jacobian matrix (output_dim x dim)
    """
    total_start_time = time.time()  # Start total time measurement

    start_time = time.time()
    # Normalize q_p_sample
    q_p_sample_norm = scaler.transform(q_p_sample.reshape(1, -1)).reshape(-1)
    normalize_time = time.time() - start_time
    print(f"Normalized q_p_sample in {normalize_time:.6f} seconds")

    start_time = time.time()
    # Compute pairwise distances between q_p_sample_norm and all training points
    dist_to_sample = np.linalg.norm(q_p_train_norm - q_p_sample_norm, axis=1)  # Shape: (num_train,)
    dist_time = time.time() - start_time
    print(f"Computed distances to sample in {dist_time:.6f} seconds")

    start_time = time.time()
    # Compute RBF values for the sample
    rbf_values = rbf_func(dist_to_sample, epsilon)  # Shape: (num_train,)
    rbf_time = time.time() - start_time
    print(f"Computed RBF values in {rbf_time:.6f} seconds")

    start_time = time.time()
    # Interpolate the new value
    f_new = rbf_values @ W_neighbors  # Shape: (output_dim,)
    interp_time = time.time() - start_time
    print(f"Interpolated new value in {interp_time:.6f} seconds")

    start_time = time.time()
    # Compute the Jacobian
    print("Computing the Jacobian matrix...")
    dim = q_p_sample_norm.shape[0]
    output_dim = W_neighbors.shape[1]
    jacobian_norm = np.zeros((output_dim, dim))

    for i in range(q_p_train_norm.shape[0]):
        q_p_i = q_p_train_norm[i]
        r_i = dist_to_sample[i]
        phi_r_i = rbf_values[i]

        if r_i > 1e-12:
            if kernel_type == 'gaussian':
                dphi_dq_p_norm = -2 * epsilon**2 * phi_r_i * (q_p_sample_norm - q_p_i)
            elif kernel_type == 'imq':
                dphi_dq_p_norm = -epsilon**2 * (phi_r_i ** 3) * (q_p_sample_norm - q_p_i)
            elif kernel_type == 'multiquadric':
                dphi_dq_p_norm = epsilon**2 * (phi_r_i ** (-1)) * (q_p_sample_norm - q_p_i)
            elif kernel_type == 'linear':
                dphi_dq_p_norm = (q_p_sample_norm - q_p_i) / r_i
            else:
                raise ValueError("Unsupported kernel type.")
        else:
            dphi_dq_p_norm = np.zeros_like(q_p_sample_norm)

        jacobian_norm += np.outer(W_neighbors[i], dphi_dq_p_norm)

    compute_jacobian_time = time.time() - start_time
    print(f"Computed Jacobian matrix in {compute_jacobian_time:.6f} seconds")

    start_time = time.time()
    # Adjust the Jacobian to account for Min-Max normalization
    scale = scaler.scale_  # This is 1 / (q_p_max - q_p_min)
    jacobian = jacobian_norm * scale[np.newaxis, :]
    adjust_jacobian_time = time.time() - start_time
    print(f"Adjusted Jacobian matrix in {adjust_jacobian_time:.6f} seconds")

    # Total time
    total_time = time.time() - total_start_time
    print(f"Total time for interpolation and Jacobian computation: {total_time:.6f} seconds")

    return f_new, jacobian

def gradient_check_pod_rbf(
    kernel_type, rbf_func, U_p, snapshot_column,
    q_p_train_norm, W_neighbors, epsilon_values, epsilon,
    scaler
):
    """
    Gradient check for the POD-RBF model using a specified RBF kernel with the global approach.
    """
    # Project the snapshot onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column  # Shape: (n_p,)
    
    # Interpolate and compute the Jacobian
    print(f"\nInterpolating and computing Jacobian for {kernel_type.capitalize()} RBF (Global Approach)...")
    f_new, jacobian = interpolate_and_compute_jacobian(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p.reshape(1, -1), epsilon, scaler
    )
    
    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Normalize
    
    # Initialize list to store errors
    errors = []
    
    print("Performing gradient checks with varying epsilon values...")
    for eps in epsilon_values:
        # Perturb q_p and compute the RBF output for the perturbed q_p
        q_p_perturbed = q_p + eps * v
        f_perturbed, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_perturbed.reshape(1, -1), epsilon, scaler
        )
        
        # Calculate the error term
        error = np.linalg.norm(f_perturbed - f_new - eps * (jacobian @ v))
        errors.append(error)
        print(f"Epsilon: {eps:.1e}, Error: {error:.3e}")
    
    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')
    
    # Add reference lines for linear (O(epsilon)) and quadratic (O(epsilon^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')
    
    plt.xlabel('Perturbation Epsilon')
    plt.ylabel('Gradient Check Error')
    plt.title(f'Gradient Check Error vs. Epsilon ({kernel_type.capitalize()} RBF, Global Approach)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Compute and print the slopes
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print(f"Slopes between consecutive points on the log-log plot for {kernel_type.capitalize()} RBF (Global):", slopes)
    
    plt.show()


if __name__ == '__main__':
    # Load the precomputed global weights and training data
    global_weights_filename = 'pod_rbf_global_model/global_weights.pkl'
    try:
        with open(global_weights_filename, 'rb') as f:
            global_data = pickle.load(f)
            W_global = global_data['W']              # Shape: (num_train x output_dim)
            q_p_train_norm = global_data['q_p_train'] # Shape: (num_train x dim)
            epsilon = global_data['epsilon']
            kernel_name = global_data['kernel_name']
        print(f"Loaded global weights from '{global_weights_filename}'.")
    except FileNotFoundError:
        print(f"File '{global_weights_filename}' not found. Please ensure it exists.")
        exit(1)

    # Load the scaler
    scaler_filename = 'pod_rbf_global_model/scaler.pkl'
    try:
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from '{scaler_filename}'.")
    except FileNotFoundError:
        print(f"Scaler file '{scaler_filename}' not found. Please ensure it exists.")
        exit(1)

    # Load U_p (primary POD basis)
    U_p_filename = 'pod_rbf_global_model/U_p.npy'
    try:
        U_p = np.load(U_p_filename)
        print(f"Loaded primary POD basis from '{U_p_filename}'.")
    except FileNotFoundError:
        print(f"Primary POD basis file '{U_p_filename}' not found. Please ensure it exists.")
        exit(1)

    # Load the snapshot data and select a specific column
    snapshot_file = '../param_snaps/mu1_5.19+mu2_0.026.npy'   # Adjust the filename as needed
    try:
        snapshot = np.load(snapshot_file)
        snapshot_column = snapshot[:, 40]
        print(f"Loaded snapshot from '{snapshot_file}'.")
    except FileNotFoundError:
        print(f"Snapshot file '{snapshot_file}' not found. Please ensure it exists.")
        exit(1)

    # Define epsilon values for the gradient check
    epsilon_values = np.logspace(-6, -1, 12)  # From 1e-6 to 1e-1

    # Perform gradient check for the specific kernel
    if kernel_name == 'gaussian':
        rbf_func = gaussian_rbf
    elif kernel_name == 'imq':
        rbf_func = imq_rbf
    elif kernel_name == 'multiquadric':
        rbf_func = mq_rbf
    elif kernel_name == 'linear':
        rbf_func = linear_rbf
    else:
        print(f"Unsupported kernel name '{kernel_name}' found in the global weights.")
        exit(1)

    print(f"\n--- Gradient Check: {kernel_name.capitalize()} RBF with Global Approach ---")
    gradient_check_pod_rbf(
        kernel_type=kernel_name,
        rbf_func=rbf_func,
        U_p=U_p,
        snapshot_column=snapshot_column,
        q_p_train_norm=q_p_train_norm,
        W_neighbors=W_global,
        epsilon_values=epsilon_values,
        epsilon=epsilon,
        scaler=scaler
    )
