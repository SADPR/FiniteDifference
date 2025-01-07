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
    # Normalize q_p_sample
    q_p_sample_norm = scaler.transform(q_p_sample.reshape(1, -1)).reshape(-1)

    # Compute pairwise distances between q_p_sample_norm and all training points
    dist_to_sample = np.linalg.norm(q_p_train_norm - q_p_sample_norm, axis=1)

    # Compute RBF values for the sample
    rbf_values = rbf_func(dist_to_sample, epsilon)

    # Interpolate the new value
    f_new = rbf_values @ W_neighbors

    # Compute the Jacobian
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

    # Adjust the Jacobian to account for Min-Max normalization
    scale = scaler.scale_  # This is 1 / (q_p_max - q_p_min)
    jacobian = jacobian_norm * scale[np.newaxis, :]

    return f_new, jacobian

def compute_finite_difference_jacobian(
    f_func, q_p_sample, dim, epsilon=1e-6
):
    """
    Compute the Jacobian using finite differences.

    Parameters:
    - f_func: function to compute interpolated values
    - q_p_sample: np.ndarray, single sample primary coordinates
    - dim: int, dimension of the input space
    - epsilon: float, step size for finite differences

    Returns:
    - fd_jacobian: np.ndarray, finite difference Jacobian (output_dim x dim)
    """
    # Evaluate the function once to determine the output size
    f_sample = f_func(q_p_sample)
    output_dim = f_sample.shape[0]  # Determine n_out

    # Initialize Jacobian with the correct shape (n_out, n_in)
    fd_jacobian = np.zeros((output_dim, dim))

    # Iterate over each input dimension
    for i in range(dim):
        # Create perturbed versions of the input vector
        q_p_plus = q_p_sample.copy()
        q_p_minus = q_p_sample.copy()
        q_p_plus[i] += epsilon
        q_p_minus[i] -= epsilon

        # Compute the function values at perturbed points
        f_plus = f_func(q_p_plus)  # Shape: (n_out,)
        f_minus = f_func(q_p_minus)  # Shape: (n_out,)

        # Compute the finite difference for this input dimension
        fd_jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return fd_jacobian


def gradient_check_pod_rbf_with_fd(
    kernel_type, rbf_func, U_p, snapshot_column,
    q_p_train_norm, W_neighbors, epsilon_values, epsilon,
    scaler
):
    """
    Gradient check for the POD-RBF model using both analytical and finite difference Jacobians.
    """
    q_p = U_p.T @ snapshot_column  # Shape: (n_p,)

    f_new, jacobian_analytical = interpolate_and_compute_jacobian(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p, epsilon, scaler
    )

    def f_func(q_p_sample):
        f_sample, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_sample, epsilon, scaler
        )
        return f_sample

    fd_jacobian = compute_finite_difference_jacobian(f_func, q_p, q_p.shape[0])

    print("Comparing Analytical vs Finite Difference Jacobians...")
    print("Analytical Jacobian:")
    print(jacobian_analytical)
    print("Finite Difference Jacobian:")
    print(fd_jacobian)
    print("Difference:")
    print(np.abs(jacobian_analytical - fd_jacobian))

    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Normalize

    # Initialize list to store errors
    errors_analytical = []
    errors_fd = []

    print("Performing gradient checks with varying epsilon values...")
    for eps in epsilon_values:
        q_p_perturbed = q_p + eps * v
        f_perturbed, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_perturbed, epsilon, scaler
        )

        error_analytical = np.linalg.norm(f_perturbed - f_new - eps * (jacobian_analytical @ v))
        error_fd = np.linalg.norm(f_perturbed - f_new - eps * (fd_jacobian @ v))
        errors_analytical.append(error_analytical)
        errors_fd.append(error_fd)
        print(f"Epsilon: {eps:.1e}, Analytical Error: {error_analytical:.3e}, FD Error: {error_fd:.3e}")

    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors_analytical, marker='o', label='Analytical Error')
    plt.loglog(epsilon_values, errors_fd, marker='x', label='Finite Difference Error')
    plt.xlabel('Perturbation Epsilon')
    plt.ylabel('Gradient Check Error')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    # Compute and print the slopes
    slopes_analytical = np.diff(np.log(errors_analytical)) / np.diff(np.log(epsilon_values))
    slopes_fd = np.diff(np.log(errors_fd)) / np.diff(np.log(epsilon_values))
    print(f"Slopes (Analytical) for {kernel_type.capitalize()} RBF: {slopes_analytical}")
    print(f"Slopes (Finite Difference) for {kernel_type.capitalize()} RBF: {slopes_fd}")


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

    print(f"\n--- Gradient Check: {kernel_name.capitalize()} RBF ---")
    gradient_check_pod_rbf_with_fd(
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
