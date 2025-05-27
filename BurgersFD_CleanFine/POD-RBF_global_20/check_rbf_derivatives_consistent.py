import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

#######################################
# RBF Kernels
#######################################
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

#######################################
# Interpolate + Jacobian (Already in Normed Space)
#######################################
def interpolate_and_compute_jacobian_normed(
    kernel_type, rbf_func, q_p_train_norm, W_neighbors,
    q_p_sample_norm, epsilon
):
    """
    Interpolate at a new *already normalized* point and compute the Jacobian 
    directly in normalized space. No call to scaler.transform or scale multiplication.

    Parameters:
    - kernel_type: str, RBF type ('gaussian','imq','mq','linear').
    - rbf_func: function, RBF kernel.
    - q_p_train_norm: np.ndarray, normalized training coords (num_train x dim).
    - W_neighbors: np.ndarray, precomputed weights (num_train x output_dim).
    - q_p_sample_norm: np.ndarray, normalized sample coords (dim,).
    - epsilon: float, RBF parameter.

    Returns:
    - f_new: np.ndarray, interpolated output in normalized RBF sense (output_dim,).
    - jacobian_norm: np.ndarray, derivative wrt normalized coords (output_dim x dim).
    """
    num_train = q_p_train_norm.shape[0]
    output_dim = W_neighbors.shape[1]
    dim = q_p_sample_norm.shape[0]

    # 1) Compute distances in normalized space
    dist_to_sample = np.linalg.norm(q_p_train_norm - q_p_sample_norm, axis=1)  # (num_train,)

    # 2) Compute RBF kernel values
    rbf_values = rbf_func(dist_to_sample, epsilon)  # (num_train,)

    # 3) Interpolate
    f_new = rbf_values @ W_neighbors  # (output_dim,)

    # 4) Compute Jacobian wrt normalized coords
    jacobian_norm = np.zeros((output_dim, dim))

    for i in range(num_train):
        r_i = dist_to_sample[i]
        phi_r_i = rbf_values[i]
        dq = (q_p_sample_norm - q_p_train_norm[i])  # shape: (dim,)

        if r_i > 1e-12:
            if kernel_type == 'gaussian':
                dphi = -2 * epsilon**2 * phi_r_i * dq
            elif kernel_type == 'imq':
                dphi = -epsilon**2 * (phi_r_i ** 3) * dq
            elif kernel_type == 'multiquadric':
                dphi = epsilon**2 * (phi_r_i ** (-1)) * dq
            elif kernel_type == 'linear':
                dphi = dq / r_i
            else:
                raise ValueError("Unsupported kernel type.")
        else:
            dphi = np.zeros_like(dq)

        jacobian_norm += np.outer(W_neighbors[i], dphi)

    return f_new, jacobian_norm


#######################################
# Gradient Check in Normalized Space
#######################################
def gradient_check_pod_rbf(
    kernel_type, rbf_func, U_p, snapshot_column,
    q_p_train_norm, W_neighbors, epsilon_values, epsilon,
    scaler
):
    """
    Gradient check for the POD-RBF model using a specified RBF kernel 
    with a purely normalized approach.

    Steps:
    1) Project snapshot -> q_p (original coords).
    2) Normalize q_p -> q_p_norm.
    3) Evaluate interpolate_and_compute_jacobian_normed in normalized space.
    4) Perturb q_p_norm by eps * v in normalized space, re-evaluate, measure error.
    """
    # 1) Project onto primary modes (original coords)
    q_p = U_p.T @ snapshot_column  # shape: (n_p,)

    # 2) Normalize once outside
    q_p_norm = scaler.transform(q_p.reshape(1, -1)).reshape(-1)

    # Evaluate the function + Jacobian in normalized coords
    print(f"\n--- Gradient Check (Normalized) for {kernel_type} RBF ---")
    f_new_norm, jacobian_norm = interpolate_and_compute_jacobian_normed(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p_norm, epsilon
    )

    # 3) Set up a random perturbation vector in normalized space
    v = np.random.randn(*q_p_norm.shape)
    v /= (np.linalg.norm(v) + 1e-12)

    errors = []

    print("Performing gradient checks with varying epsilon values in normalized coords...")
    for eps in epsilon_values:
        # 4) Perturb in normalized space
        q_p_norm_perturbed = q_p_norm + eps * v

        # Recompute the function for the perturbed point in normalized coords
        f_perturbed_norm, _ = interpolate_and_compute_jacobian_normed(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_norm_perturbed, epsilon
        )

        # Compare to linear approximation
        error = np.linalg.norm(f_perturbed_norm - f_new_norm - eps * (jacobian_norm @ v))
        errors.append(error)
        print(f"Epsilon: {eps:.1e}, Error: {error:.3e}")

    # Plot the errors
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')

    # Reference lines
    plt.loglog(epsilon_values, epsilon_values*errors[0]/epsilon_values[0], 'r--', label='O(eps) Reference')
    plt.loglog(epsilon_values, (epsilon_values**2)*(errors[0]/(epsilon_values[0]**2)), 'g--', label='O(eps^2) Reference')

    plt.xlabel('Perturbation Epsilon')
    plt.ylabel('Gradient Check Error')
    plt.title(f'Gradient Check Error vs. Epsilon ({kernel_type.capitalize()} RBF, Normalized Space)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print(f"Slopes on log-log for {kernel_type.capitalize()} RBF (Normalized): {slopes}")

    plt.show()


#######################################
# Main
#######################################
if __name__ == '__main__':
    global_weights_filename = 'pod_rbf_global_model/global_weights.pkl'
    try:
        with open(global_weights_filename, 'rb') as f:
            global_data = pickle.load(f)
            W_global = global_data['W']              # (num_train x output_dim)
            q_p_train_norm = global_data['q_p_train'] # (num_train x dim) normalized
            epsilon = global_data['epsilon']
            kernel_name = global_data['kernel_name']
        print(f"Loaded global weights from '{global_weights_filename}'.")
    except FileNotFoundError:
        print(f"File '{global_weights_filename}' not found.")
        exit(1)

    scaler_filename = 'pod_rbf_global_model/scaler.pkl'
    try:
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from '{scaler_filename}'.")
    except FileNotFoundError:
        print(f"Scaler file '{scaler_filename}' not found.")
        exit(1)

    U_p_filename = 'pod_rbf_global_model/U_p.npy'
    try:
        U_p = np.load(U_p_filename)
        print(f"Loaded primary POD basis from '{U_p_filename}'.")
    except FileNotFoundError:
        print(f"Primary POD basis file '{U_p_filename}' not found.")
        exit(1)

    snapshot_file = '../param_snaps/mu1_4.75+mu2_0.02.npy'
    try:
        snapshot = np.load(snapshot_file)
        snapshot_column = snapshot[:, 100]
        print(f"Loaded snapshot from '{snapshot_file}'.")
    except FileNotFoundError:
        print(f"Snapshot file '{snapshot_file}' not found.")
        exit(1)

    # RBF kernel mapping
    rbf_map = {
        'gaussian': gaussian_rbf,
        'imq': imq_rbf,
        'multiquadric': mq_rbf,
        'linear': linear_rbf,
    }
    if kernel_name not in rbf_map:
        print(f"Unsupported kernel '{kernel_name}' found.")
        exit(1)

    # Epsilon range for gradient check
    epsilon_values = np.logspace(-6, -1, 12)

    rbf_func = rbf_map[kernel_name]

    # Perform gradient check in normalized space
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
