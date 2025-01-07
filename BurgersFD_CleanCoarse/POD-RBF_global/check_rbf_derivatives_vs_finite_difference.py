import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import MinMaxScaler

# Define the RBF kernels
def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    """Inverse Multiquadric (IMQ) RBF kernel function."""
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def multiquadric_rbf(r, epsilon):
    """Multiquadric (MQ) RBF kernel function."""
    return np.sqrt(1 + (epsilon * r) ** 2)

def linear_rbf(r, epsilon):
    """Linear RBF kernel function."""
    return r

def matern52_rbf(r, epsilon):
    """
    Matern 5/2 kernel.
    φ(r) = (1 + sqrt(5)*εr + (5*(εr)^2)/3) * exp(-sqrt(5)*εr)
    """
    sqrt5 = np.sqrt(5.0)
    scaled_r = epsilon * r
    tmp = sqrt5 * scaled_r
    return (1.0 + tmp + (tmp**2)/3.0) * np.exp(-tmp)

def compact_bump_rbf(r, epsilon):
    """
    Compactly-supported 'bump' function kernel.
    φ(r) = exp(1 / ((ε*r)^2 - 1)) for ε*r < 1, else 0.
    """
    scaled_r = epsilon * r
    phi = np.zeros_like(scaled_r)
    inside_mask = scaled_r < 1.0
    # For points where scaled_r < 1, define the bump
    phi[inside_mask] = np.exp(1.0 / (scaled_r[inside_mask]**2 - 1.0))
    return phi

def thin_plate_spline_rbf(r, epsilon):
    """
    Thin Plate Spline (TPS) kernel.
    φ(r) = (epsilon * r)^2 * log(epsilon * r + 1e-15) for r > 0
    """
    scaled_r = epsilon * r
    # Avoid log(0) by adding a tiny offset
    scaled_r = np.where(scaled_r < 1e-15, 1e-15, scaled_r)
    return (scaled_r ** 2) * np.log(scaled_r)

def wendland_c2_rbf(r, epsilon):
    """
    Wendland C2 kernel (compactly supported).
    φ(r) = (1 - epsilon*r)^4 * (4*epsilon*r + 1) for epsilon*r < 1, else 0
    """
    scaled_r = epsilon * r
    phi = np.zeros_like(scaled_r)
    inside = scaled_r < 1.0
    tmp = 1.0 - scaled_r[inside]
    phi[inside] = (tmp ** 4) * (4.0 * scaled_r[inside] + 1.0)
    return phi

def rational_quadratic_rbf(r, epsilon, alpha=1.0):
    """
    Rational Quadratic kernel:
    φ(r) = (1 + (epsilon*r)^2 / (2*alpha))^(-alpha)
    """
    scaled_r_sq = (epsilon * r) ** 2
    return (1.0 + scaled_r_sq / (2.0 * alpha)) ** (-alpha)

def cubic_rbf(r, epsilon):
    """
    Cubic RBF:
    φ(r) = (epsilon * r)^3
    """
    return (epsilon * r) ** 3

def quintic_rbf(r, epsilon):
    """
    Quintic RBF:
    φ(r) = (epsilon * r)^5
    """
    return (epsilon * r) ** 5

#############################
# Combined Dictionary of RBF Kernels
#############################

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'multiquadric': multiquadric_rbf,
    'linear': linear_rbf,
    'matern52': matern52_rbf,
    'bump': compact_bump_rbf,
    'thin_plate_spline': thin_plate_spline_rbf,
    'wendland_c2': wendland_c2_rbf,
    'rational_quadratic': rational_quadratic_rbf,
    'cubic': cubic_rbf,
    'quintic': quintic_rbf
}

def interpolate_and_compute_jacobian(
    kernel_type, rbf_func, q_p_train_norm, W_neighbors,
    q_p_sample, epsilon, scaler
):
    """
    Interpolate at a new point and compute the Jacobian using the global approach.

    Parameters:
    - kernel_type: str, type of RBF kernel ('gaussian', 'imq', 'mq', etc.)
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
            delta = q_p_sample_norm - q_p_i
            if kernel_type == 'gaussian':
                dphi_dq_p_norm = -2 * epsilon**2 * phi_r_i * delta
            elif kernel_type == 'imq':
                dphi_dq_p_norm = -epsilon**2 * (phi_r_i ** 3) * delta
            elif kernel_type == 'multiquadric':
                dphi_dq_p_norm = epsilon**2 * (phi_r_i ** (-1)) * delta
            elif kernel_type == 'linear':
                dphi_dq_p_norm = delta / r_i
            elif kernel_type == 'matern52':
                sqrt5 = np.sqrt(5.0)
                dphi_dq_p_norm = (
                    (-sqrt5 * epsilon) * phi_r_i +
                    (-sqrt5**2 * epsilon**2) * phi_r_i * delta +
                    (-5.0 * epsilon**2 / 3.0) * phi_r_i * delta
                )
            elif kernel_type == 'bump':
                # The derivative of the bump function
                dphi_dq_p_norm = (
                    -2 * epsilon**2 * (phi_r_i ** 3) * delta
                )
            elif kernel_type == 'thin_plate_spline':
                dphi_dq_p_norm = 2 * epsilon**2 * delta * np.log(epsilon * r_i + 1e-15) + \
                                  (epsilon * r_i) * delta / (epsilon * r_i + 1e-15)
            elif kernel_type == 'wendland_c2':
                dphi_dq_p_norm = -4 * (1 - epsilon * r_i) ** 3 * (4 * epsilon * r_i + 1) * epsilon * delta
            elif kernel_type == 'rational_quadratic':
                alpha = 1.0  # You can parameterize alpha if needed
                dphi_dq_p_norm = - (2 * alpha * epsilon**2 * r_i**2) / (2 * alpha + (epsilon * r_i)**2) * \
                                   (1 + (epsilon * r_i)**2 / (2 * alpha)) ** (-alpha - 1) * delta
            elif kernel_type == 'cubic':
                dphi_dq_p_norm = 3 * (epsilon * r_i) ** 2 * epsilon * delta
            elif kernel_type == 'quintic':
                dphi_dq_p_norm = 5 * (epsilon * r_i) ** 4 * epsilon * delta
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

def finite_difference_jacobian(kernel_type, rbf_func, q_p_train_norm, W_neighbors,
                               q_p_sample, epsilon, scaler, h=1e-6):
    """
    Compute the Jacobian using finite differences (central differences).

    Parameters:
    - Same as interpolate_and_compute_jacobian
    - h: float, small perturbation size

    Returns:
    - jacobian_fd: np.ndarray, finite differences Jacobian (output_dim x dim)
    """
    dim = q_p_sample.shape[1]
    output_dim = W_neighbors.shape[1]
    jacobian_fd = np.zeros((output_dim, dim))

    for d in range(dim):
        perturb = np.zeros_like(q_p_sample)
        perturb[0, d] = h

        # Compute f(x + h * e_d)
        f_plus, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_sample + perturb, epsilon, scaler
        )

        # Compute f(x - h * e_d)
        f_minus, _ = interpolate_and_compute_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p_sample - perturb, epsilon, scaler
        )

        # Central difference approximation
        df = (f_plus - f_minus) / (2 * h)
        jacobian_fd[:, d] = df

    return jacobian_fd

def gradient_check_pod_rbf(
    kernel_type, rbf_func, U_p, snapshot_column,
    q_p_train_norm, W_neighbors, epsilon_values, epsilon,
    scaler
):
    """
    Gradient check for the POD-RBF model using a specified RBF kernel with the global approach.
    Compares analytic Jacobian with finite differences Jacobian.
    """
    # Project the snapshot onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column  # Shape: (n_p,)

    # Reshape for consistency
    q_p = q_p.reshape(1, -1)

    # Interpolate and compute the analytic Jacobian
    print(f"\nInterpolating and computing Analytic Jacobian for {kernel_type.capitalize()} RBF (Global Approach)...")
    f_new, jacobian = interpolate_and_compute_jacobian(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p, epsilon, scaler
    )

    # Compute the finite differences Jacobian
    print("Computing Finite Differences Jacobian...")
    jacobian_fd = finite_difference_jacobian(
        kernel_type, rbf_func, q_p_train_norm, W_neighbors,
        q_p, epsilon, scaler, h=1e-6
    )

    # Compare the two Jacobians
    comparison = jacobian - jacobian_fd
    error_norm = np.linalg.norm(comparison)
    relative_error = error_norm / np.linalg.norm(jacobian_fd)
    print(f"Analytic Jacobian Norm: {np.linalg.norm(jacobian)}")
    print(f"Finite Differences Jacobian Norm: {np.linalg.norm(jacobian_fd)}")
    print(f"Difference Norm: {error_norm}")
    print(f"Relative Error: {relative_error:.3e}")

    # Generate a heatmap comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(jacobian, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Analytic Jacobian')

    plt.subplot(1, 3, 2)
    plt.imshow(jacobian_fd, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Finite Differences Jacobian')

    plt.subplot(1, 3, 3)
    plt.imshow(comparison, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Analytic - Finite Differences')

    plt.tight_layout()
    plt.show()

    # Initialize list to store errors for gradient checks with varying epsilon values
    errors = []

    print("Performing gradient checks with varying perturbation sizes (h)...")
    for h in epsilon_values:
        # Compute finite differences Jacobian with perturbation h
        jacobian_fd_h = finite_difference_jacobian(
            kernel_type, rbf_func, q_p_train_norm, W_neighbors,
            q_p, epsilon, scaler, h=h
        )

        # Compute the difference between analytic and finite differences
        diff = jacobian - jacobian_fd_h
        error = np.linalg.norm(diff)
        errors.append(error)
        print(f"Perturbation h: {h:.1e}, Difference Norm: {error:.3e}")

    # Plot the errors against perturbation size
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Difference Norm')

    # Reference lines for different orders
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($h$)')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($h^2$)')

    plt.xlabel('Perturbation Size h')
    plt.ylabel('Difference Norm ||J_analytic - J_fd||')
    plt.title(f'Gradient Check Error vs. Perturbation Size ({kernel_type.capitalize()} RBF)')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # Compute and print the slopes
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print(f"Slopes between consecutive points on the log-log plot for {kernel_type.capitalize()} RBF:", slopes)

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

    # Define perturbation sizes for gradient checks (h)
    epsilon_values = np.logspace(-6, -1, 12)  # From 1e-6 to 1e-1

    # Perform gradient check for the specific kernel
    if kernel_name in rbf_kernels:
        rbf_func = rbf_kernels[kernel_name]
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
