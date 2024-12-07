import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle
import time
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' depending on your system



def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def interpolate_and_compute_jacobian_gaussian_with_normalization(kdtree, q_p_train_norm, q_s_train, q_p_sample, epsilon, neighbors, scaler):
    """
    Interpolate at new points and compute the Jacobian on the fly using nearest neighbors and Gaussian RBF,
    accounting for Min-Max normalization.
    """
    start_time = time.time()
    
    # Normalize q_p_sample
    q_p_sample_norm = scaler.transform(q_p_sample.reshape(1, -1)).reshape(-1)
    
    # Find the nearest neighbors for the normalized sample point
    dist, idx = kdtree.query(q_p_sample_norm.reshape(1, -1), k=neighbors)
    query_time = time.time()
    print(f"KDTree query took: {query_time - start_time:.6f} seconds")
    
    # Extract the neighbor points and their corresponding outputs
    X_neighbors = q_p_train_norm[idx].reshape(neighbors, -1)  # Normalized
    Y_neighbors = q_s_train[idx].reshape(neighbors, -1)
    extract_time = time.time()
    print(f"Extracting neighbors took: {extract_time - query_time:.6f} seconds")
    
    # Compute pairwise distances between neighbors
    dists_neighbors = np.linalg.norm(X_neighbors[:, None, :] - X_neighbors[None, :, :], axis=-1)
    dist_time = time.time()
    print(f"Distance calculations took: {dist_time - extract_time:.6f} seconds")
    
    # Compute the Gaussian RBF matrix for the neighbors (Phi matrix)
    Phi_neighbors = gaussian_rbf(dists_neighbors, epsilon)
    cond_number = np.linalg.cond(Phi_neighbors)
    print(f"Condition number of Phi: {cond_number}")
    Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization
    cond_number = np.linalg.cond(Phi_neighbors)
    print(f"Condition number of Phi: {cond_number}")
    rbf_matrix_time = time.time()
    print(f"RBF matrix computation took: {rbf_matrix_time - dist_time:.6f} seconds")
    
    # Solve for the weights
    W_neighbors = np.linalg.solve(Phi_neighbors, Y_neighbors)
    solve_time = time.time()
    print(f"Solving the linear system took: {solve_time - rbf_matrix_time:.6f} seconds")
    
    # Compute RBF values between q_p_sample_norm and its neighbors
    dist_to_sample = np.linalg.norm(X_neighbors - q_p_sample_norm, axis=1)
    rbf_values = gaussian_rbf(dist_to_sample, epsilon)
    rbf_eval_time = time.time()
    print(f"RBF evaluation for new point took: {rbf_eval_time - solve_time:.6f} seconds")
    
    # Interpolate the new value
    f_new = rbf_values @ W_neighbors
    interp_time = time.time()
    print(f"Interpolation step took: {interp_time - rbf_eval_time:.6f} seconds")
    
    # Compute the Jacobian with respect to normalized q_p
    input_dim = q_p_sample_norm.shape[0]
    output_dim = Y_neighbors.shape[1]
    jacobian_norm = np.zeros((output_dim, input_dim))
    
    for i in range(neighbors):
        q_p_i = X_neighbors[i]
        r_i = dist_to_sample[i]
        phi_r_i = rbf_values[i]
        
        if r_i > 1e-12:
            dphi_dq_p_norm = -2 * epsilon**2 * (q_p_sample_norm - q_p_i) * phi_r_i
        else:
            dphi_dq_p_norm = np.zeros_like(q_p_sample_norm)
        
        jacobian_norm += np.outer(W_neighbors[i], dphi_dq_p_norm)
    
    jacobian_time = time.time()
    print(f"Jacobian computation took: {jacobian_time - interp_time:.6f} seconds")
    
    # Adjust the Jacobian to account for Min-Max normalization
    # Extract scaling factors from the scaler
    scale = scaler.scale_  # This is 1 / (q_p_max - q_p_min)
    # Adjust the Jacobian
    jacobian = jacobian_norm * scale[np.newaxis, :]
    
    return f_new, jacobian

def gradient_check_pod_rbf_gaussian_with_normalization(U_p, snapshot_column, q_p_train_norm, q_s_train, epsilon_values, epsilon, kdtree, neighbors, scaler):
    """
    Gradient check for the POD-RBF model using the Gaussian RBF kernel with normalization.
    """
    # Project the snapshot_column onto the primary POD basis to get q_p
    q_p = U_p.T @ snapshot_column  # Shape: (n_p,)
    
    # Interpolate and compute the Jacobian dynamically
    f_new, jacobian = interpolate_and_compute_jacobian_gaussian_with_normalization(
        kdtree, q_p_train_norm, q_s_train, q_p.reshape(1, -1), epsilon, neighbors, scaler
    )
    
    # Generate a random unit vector v of the same size as q_p
    v = np.random.randn(*q_p.shape)
    v /= np.linalg.norm(v) + 1e-10  # Normalize
    
    # Initialize list to store errors
    errors = []
    
    for eps in epsilon_values:
        # Perturb q_p and compute the RBF output for the perturbed q_p
        q_p_perturbed = q_p + eps * v
        f_perturbed, _ = interpolate_and_compute_jacobian_gaussian_with_normalization(
            kdtree, q_p_train_norm, q_s_train, q_p_perturbed.reshape(1, -1), epsilon, neighbors, scaler
        )
        
        # Calculate the error term
        error = np.linalg.norm(f_perturbed - f_new - eps * (jacobian @ v))
        errors.append(error)
    
    # Plot the errors against epsilon
    plt.figure(figsize=(8, 6))
    plt.loglog(epsilon_values, errors, marker='o', label='Computed Error')
    
    # Add reference lines for linear (O(epsilon)) and quadratic (O(epsilon^2)) behavior
    plt.loglog(epsilon_values, epsilon_values * errors[0] / epsilon_values[0], 'r--', label=r'O($\epsilon$) Reference')
    plt.loglog(epsilon_values, epsilon_values**2 * errors[0] / epsilon_values[0]**2, 'g--', label=r'O($\epsilon^2$) Reference')
    
    plt.xlabel('epsilon')
    plt.ylabel('Error')
    plt.title('Gradient Check Error vs. Epsilon (Gaussian RBF with Normalization)')
    plt.grid(True)
    plt.legend()
    
    # Compute and print the slopes
    slopes = np.diff(np.log(errors)) / np.diff(np.log(epsilon_values))
    print("Slopes between consecutive points on the log-log plot:", slopes)
    
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import pickle
    from sklearn.preprocessing import MinMaxScaler
    from scipy.spatial import KDTree
    
    # Load the precomputed training data and scaler
    with open('modes/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        q_p_train_norm = data['q_p']  # Normalized q_p_train
        q_s_train = data['q_s']
    
    # Build the KDTree using normalized q_p_train
    kdtree = KDTree(q_p_train_norm)
    
    # Load the scaler
    with open('modes/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load the snapshot data and select a specific column
    snapshot_file = 'param_snaps/mu1_5.19+mu2_0.026.npy'   # Adjust the filename
    snapshot = np.load(snapshot_file)
    snapshot_column = snapshot[:, 100]
    
    # Load U_p (primary POD basis)
    U_p = np.load('modes/U_p.npy')
    
    # Define epsilon values for the gradient check
    epsilon_values = np.logspace(-6, -1, 12)
    
    # Set RBF parameters
    epsilon = 1
    neighbors = 5
    
    # Perform gradient check
    gradient_check_pod_rbf_gaussian_with_normalization(
        U_p, snapshot_column, q_p_train_norm, q_s_train, epsilon_values, epsilon, kdtree, neighbors, scaler
    )


