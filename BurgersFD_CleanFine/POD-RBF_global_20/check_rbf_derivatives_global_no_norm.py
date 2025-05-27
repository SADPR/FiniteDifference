# check_rbf_derivatives_global_no_norm.py

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import sys

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
    Interpolate at a new point and compute the Jacobian using the global approach,
    but with NO normalization steps.

    Parameters:
    - kernel_type: str, type of RBF kernel ('gaussian', 'imq', 'mq', 'linear')
    - rbf_func: function, RBF function corresponding to the kernel_type
    - q_p_train_norm: np.ndarray, primary training coordinates (num_train x dim) (unscaled now)
    - W_neighbors: np.ndarray, precomputed weights (num_train x output_dim)
    - q_p_sample: np.ndarray, single sample primary coordinates (1 x dim)
    - epsilon: float, RBF parameter
    - scaler: (IGNORED) MinMaxScaler object used for normalization in original code. Unused here.

    Returns:
    - f_new: np.ndarray, interpolated output for the sample (output_dim,)
    - jacobian: np.ndarray, Jacobian matrix (output_dim x dim)
    """
    total_start_time = time.time()

    start_time = time.time()
    # Instead of normalizing, just reshape the sample to 1D
    q_p_sample_norm = q_p_sample.reshape(-1)
    print(f"Skipped normalization. q_p_sample shape: {q_p_sample_norm.shape}")
    normalize_time = time.time() - start_time

    start_time = time.time()
    # Compute pairwise distances between q_p_sample_norm and all training points
    dist_to_sample = np.linalg.norm(q_p_train_norm - q_p_sample_norm, axis=1)  # (num_train,)
    dist_time = time.time() - start_time
    print(f"Computed distances to sample in {dist_time:.6f} seconds")

    start_time = time.time()
    # Compute RBF values for the sample
    rbf_values = rbf_func(dist_to_sample, epsilon)  # shape: (num_train,)
    rbf_time = time.time() - start_time
    print(f"Computed RBF values in {rbf_time:.6f} seconds")

    start_time = time.time()
    # Interpolate the new value
    f_new = rbf_values @ W_neighbors  # shape: (output_dim,)
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

    # In the original script, we do:
    #   scale = scaler.scale_
    #   jacobian = jacobian_norm * scale[np.newaxis, :]
    # For no normalization, skip that. Just use jacobian_norm directly.
    jacobian = jacobian_norm

    # Summaries
    adjust_jacobian_time = 0.0
    print(f"Skipped adjusting Jacobian for scaling. Using jacobian_norm directly.")

    total_time = time.time() - total_start_time
    print(f"Total time for interpolation and Jacobian computation: {total_time:.6f} seconds")

    return f_new, jacobian

def gradient_check_pod_rbf(
    kernel_type, rbf_func, U_p, snapshot_column,
    q_p_train_norm, W_neighbors, epsilon_values, epsilon,
    scaler
):
    """
    Gradient check for the POD-RBF model using a specified RBF kernel with the global approach,
    but skipping all normalization logic.

    We keep the function name + signature the same for compatibility.
    """
    # Project the snapshot onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column  # shape: (n_p,)

    # Interpolate and compute the Jacobian
    print(f"\nInterpolating and computing Jacobian for {kernel_type.capitalize()} RBF (No Normalization Approach)...")
    f_new, jacobian = interpolate_and_compute_jacobian(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p.reshape(1, -1), epsilon, scaler
    )
    
    print(np.linalg.norm(jacobian))
    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Normalize

    # Initialize list to store errors
    errors = []

    print("Performing gradient checks with varying epsilon values...")
    for eps_perturb in epsilon_values:
        # Perturb q_p and compute the RBF output for the perturbed q_p
        q_p_perturbed = q_p + eps_perturb * v
        f_perturbed, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_perturbed.reshape(1, -1), epsilon, scaler
        )

        # Calculate the error term
        error = np.linalg.norm(f_perturbed - f_new - eps_perturb * (jacobian @ v))
        errors.append(error)
        print(f"Epsilon: {eps_perturb:.1e}, Error: {error:.3e}")

    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')

    # Add reference lines for linear (O(eps)) and quadratic (O(eps^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')

    plt.xlabel('Perturbation Epsilon')
    plt.ylabel('Gradient Check Error')
    plt.title(f'Gradient Check Error vs. Epsilon ({kernel_type.capitalize()} RBF, No Norm)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Compute and print the slopes
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print(f"Slopes between consecutive points on the log-log plot for {kernel_type.capitalize()} RBF (No Norm):", slopes)

    plt.show()

if __name__ == '__main__':
    global_weights_filename = 'pod_rbf_global_model/global_weights.pkl'
    try:
        with open(global_weights_filename, 'rb') as f:
            global_data = pickle.load(f)
            W_global = global_data['W']             # shape: (num_train x output_dim)
            q_p_train_norm = global_data['q_p_train']  # shape: (num_train x dim), now unscaled
            epsilon = global_data['epsilon']
            kernel_name = global_data['kernel_name']
        print(f"Loaded global weights from '{global_weights_filename}'.")
    except FileNotFoundError:
        print(f"File '{global_weights_filename}' not found.")
        sys.exit(1)

    # We still load the scaler but do not use it for actual transformations or scale multiplication
    scaler_filename = 'pod_rbf_global_model/scaler.pkl'
    try:
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from '{scaler_filename}', but we won't apply it.")
    except FileNotFoundError:
        print(f"Scaler file '{scaler_filename}' not found. Proceeding without a scaler.")
        scaler = None

    # Load U_p (primary POD basis)
    U_p_filename = 'pod_rbf_global_model/U_p.npy'
    try:
        U_p = np.load(U_p_filename)
        print(f"Loaded primary POD basis from '{U_p_filename}'.")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Load a snapshot for derivative check
    snapshot_file = '../param_snaps/mu1_4.75+mu2_0.02.npy'
    try:
        snapshot = np.load(snapshot_file)
        snapshot_column = snapshot[:, 400]  # pick time step
        print(f"Loaded snapshot from '{snapshot_file}'.")
    except FileNotFoundError:
        print(f"Snapshot file '{snapshot_file}' not found.")
        sys.exit(1)

    # Epsilon steps for gradient check
    epsilon_values = np.logspace(-12, -1, 24)

    # RBF kernel selection
    if kernel_name == 'gaussian':
        rbf_func = gaussian_rbf
    elif kernel_name == 'imq':
        rbf_func = imq_rbf
    elif kernel_name == 'multiquadric':
        rbf_func = mq_rbf
    elif kernel_name == 'linear':
        rbf_func = linear_rbf
    else:
        print(f"Unsupported kernel name '{kernel_name}'.")
        sys.exit(1)

    print(f"\n--- Gradient Check: {kernel_name.capitalize()} RBF (No Norm) ---")
    gradient_check_pod_rbf(
        kernel_type=kernel_name,
        rbf_func=rbf_func,
        U_p=U_p,
        snapshot_column=snapshot_column,
        q_p_train_norm=q_p_train_norm,
        W_neighbors=W_global,
        epsilon_values=epsilon_values,
        epsilon=epsilon,
        scaler=scaler  # still pass it, but we skip using it
    )
