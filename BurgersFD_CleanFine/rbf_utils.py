import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree
import time

class RBFUtils:

    # Define RBF Kernels
    @staticmethod
    def gaussian_rbf(r, epsilon):
        """Gaussian RBF kernel function."""
        return np.exp(-(epsilon * r) ** 2)

    @staticmethod
    def inverse_multiquadric_rbf(r, epsilon):
        """Inverse Multiquadric RBF kernel function."""
        return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

    @staticmethod
    def linear_rbf(r, epsilon):
        """Linear RBF kernel function."""
        return r
    
    @staticmethod
    def multiquadric_rbf(r, epsilon):
        """Multiquadric RBF kernel function."""
        return np.sqrt(1 + (epsilon * r) ** 2)
    
    @staticmethod
    def compute_rbf_jacobian_nearest_neighbors_dynamic_gaussian(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, scaler, echo_level=0):
        """
        Compute the Jacobian of the RBF interpolation with respect to q_p using nearest neighbors dynamically,
        adjusted for normalization.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Normalized training data for principal modes, shape (N_train, n_p).
        - q_s_train: Training data for secondary modes, shape (N_train, n_s).
        - q_p_sample: The input sample point (unnormalized reduced coordinates, q_p), shape (n_p,).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.
        - scaler: The StandardScaler used to normalize the data.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p, shape (n_s, n_p).
        """
        start_time = time.time()

        # Step 1: Normalize q_p_sample
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))

        # Query the KDTree with normalized q_p_sample
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)

        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - start_time:.6f} seconds")

        # Step 2: Extract the neighbor points and corresponding secondary modes
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)  # Already normalized
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)

        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - start_time:.6f} seconds")

        # Step 3: Compute pairwise distances between neighbors
        dists_neighbors = squareform(pdist(q_p_neighbors))  # Shape: (k, k)

        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - start_time:.6f} seconds")

        # Step 4: Compute the RBF kernel matrix Psi
        Psi = RBFUtils.gaussian_rbf(dists_neighbors, epsilon)
        # Regularization for numerical stability
        Psi += np.eye(neighbors) * 1e-8

        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - start_time:.6f} seconds")

        # Step 5: Solve for W_neighbors
        W_neighbors = np.linalg.solve(Psi, q_s_neighbors)  # Shape: (k, n_s)

        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - start_time:.6f} seconds")

        # Step 6: Compute RBF kernel values between q_p_sample and its neighbors
        rbf_values = RBFUtils.gaussian_rbf(dist.flatten(), epsilon)  # Shape: (k,)

        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - start_time:.6f} seconds")

        # Step 7: Compute the Jacobian using vectorized operations
        # Compute the difference matrix D using normalized data
        D = q_p_sample_normalized - q_p_neighbors  # Shape: (k, n_p)

        # Compute the weighted differences
        weighted_D = (rbf_values[:, np.newaxis]) * D  # Shape: (k, n_p)

        # Compute the Jacobian w.r.t. normalized q_p
        jacobian_normalized = -2 * epsilon**2 * W_neighbors.T @ weighted_D  # Shape: (n_s, n_p)

        # Adjust the Jacobian to account for the normalization (chain rule)
        scale = scaler.scale_

        # Multiply each column of the Jacobian by scale_inv
        jacobian = jacobian_normalized * scale[np.newaxis, :]  # Broadcasting over columns

        if echo_level >= 1:
            print(f"Time to compute Jacobian: {time.time() - start_time:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_nearest_neighbors_dynamic_imq(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, scaler, echo_level=0):
        """
        Compute the Jacobian of the Inverse Multiquadric RBF interpolation with respect to q_p using nearest neighbors dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes (primary reduced coordinates), shape (N_train, n_p).
        - q_s_train: Training data for secondary modes (secondary reduced coordinates), shape (N_train, n_s).
        - q_p_sample: The input sample point (unnormalized reduced coordinates, q_p), shape (n_p,).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use (k).
        - scaler: The scaler used to normalize the data (e.g., StandardScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p, shape (n_s, n_p).
        """
        start_time = time.time()

        # Step 1: Normalize q_p_sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))
        if echo_level >= 1:
            print(f"Time to normalize q_p_sample: {time.time() - t0:.6f} seconds")

        # Query the KDTree with normalized q_p_sample
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 2: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)  # Normalized training data
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)  # Corresponding secondary modes
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 3: Compute pairwise distances between neighbors
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))  # Shape: (k, k)
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 4: Compute the Inverse Multiquadric RBF kernel matrix
        t0 = time.time()
        Psi = RBFUtils.inverse_multiquadric_rbf(dists_neighbors, epsilon)  # Shape: (k, k)
        Psi += np.eye(neighbors) * 1e-8  # Regularization for numerical stability
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 5: Solve for the weights
        t0 = time.time()
        W_neighbors = np.linalg.solve(Psi, q_s_neighbors)  # Shape: (k, n_s)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 6: Compute RBF kernel values between q_p_sample and its neighbors
        t0 = time.time()
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist.flatten(), epsilon)  # Shape: (k,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 7: Compute the Jacobian
        t0 = time.time()
        D = q_p_sample_normalized - q_p_neighbors  # Shape: (k, n_p)
        weighted_D = (rbf_values[:, np.newaxis]) * D  # Shape: (k, n_p)
        broadcasting_term = (1 + (epsilon * dist.flatten()) ** 2) ** (-3/2)
        broadcasting_term = broadcasting_term[:, np.newaxis]  # Shape: (k, 1)

        jacobian_normalized = -epsilon**2 * W_neighbors.T @ (weighted_D * broadcasting_term)  # Shape: (n_s, n_p)

        # Adjust the Jacobian for normalization
        scale = scaler.scale_
        jacobian = jacobian_normalized * scale[np.newaxis, :]  # Broadcasting over columns
        if echo_level >= 1:
            print(f"Time to compute Jacobian: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_nearest_neighbors_dynamic_linear(kdtree, q_p_train_norm, q_s_train, q_p_sample, epsilon, neighbors, scaler, echo_level=0):
        """
        Compute the Jacobian of the linear RBF interpolation with respect to q_p using nearest neighbors dynamically, 
        accounting for Min-Max normalization.
        
        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train_norm: Normalized training data for principal modes (primary reduced coordinates), shape (N_train, n_p).
        - q_s_train: Training data for secondary modes (secondary reduced coordinates), shape (N_train, n_s).
        - q_p_sample: The input sample point (unnormalized reduced coordinates, q_p), shape (n_p,).
        - epsilon: Width parameter for the linear RBF kernel (not used in linear but kept for consistency).
        - neighbors: Number of nearest neighbors to use (k).
        - scaler: The Min-Max scaler used to normalize the data.
        - echo_level: Level of verbosity for printing timing information (default: 0).
        
        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p, shape (n_s, n_p).
        """

        start_time = time.time()
        
        # Normalize q_p_sample
        q_p_sample_norm = scaler.transform(q_p_sample.reshape(1, -1)).reshape(-1)
        
        # Step 1: Find the nearest neighbors in the normalized space
        t0 = time.time()
        dist, idx = kdtree.query(q_p_sample_norm.reshape(1, -1), k=neighbors)
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 2: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train_norm[idx].reshape(neighbors, -1)  # Normalized neighbors
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)       # Secondary modes for neighbors
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 3: Compute pairwise distances between neighbors to form the RBF kernel matrix Psi
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))  # Shape: (k, k)
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 4: Compute the RBF kernel matrix Psi (linear kernel)
        t0 = time.time()
        Psi = dists_neighbors  # Linear RBF: psi(r) = r
        Psi += np.eye(neighbors) * 1e-8  # Regularization for numerical stability
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 5: Compute the weights W_neighbors
        t0 = time.time()
        W_neighbors = np.linalg.solve(Psi, q_s_neighbors)  # Shape: (k, n_s)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 6: Compute RBF values between q_p_sample_norm and its neighbors (psi vector)
        t0 = time.time()
        rbf_values = dist.flatten()  # For linear RBF, psi_i = r_i
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 7: Compute the Jacobian using vectorized operations
        t0 = time.time()

        # Compute the difference matrix D and the norm r for gradient calculation
        D = q_p_sample_norm - q_p_neighbors  # Shape: (k, n_p)
        r = dist.flatten()                   # Shape: (k,)

        # Calculate r_inv, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            r_inv = np.where(r != 0, 1.0 / r, 0.0)  # Shape: (k,)

        # Gradient of psi in the normalized space
        grad_psi_norm = D * r_inv[:, np.newaxis]  # Shape: (k, n_p)

        # Compute the Jacobian with respect to the normalized coordinates
        jacobian_norm = W_neighbors.T @ grad_psi_norm  # Shape: (n_s, n_p)

        # Adjust the Jacobian to account for Min-Max normalization
        scale = scaler.scale_  # Min-Max scaling factor: 1 / (q_p_max - q_p_min)
        jacobian = jacobian_norm * scale[np.newaxis, :]  # Broadcasting over columns

        if echo_level >= 1:
            print(f"Time to compute Jacobian: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_nearest_neighbors_dynamic_multiquadric(kdtree, q_p_train, q_s_train, q_p_sample, epsilon, neighbors, scaler, echo_level=0):
        """
        Compute the Jacobian of the Multiquadric RBF interpolation with respect to q_p using nearest neighbors dynamically,
        adjusted for normalization.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Normalized training data for principal modes, shape (N_train, n_p).
        - q_s_train: Training data for secondary modes, shape (N_train, n_s).
        - q_p_sample: The input sample point (unnormalized reduced coordinates, q_p), shape (n_p,).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.
        - scaler: The StandardScaler used to normalize the data.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - jacobian: The Jacobian matrix of the RBF's output with respect to q_p, shape (n_s, n_p).
        """
        start_time = time.time()

        # Step 1: Normalize q_p_sample
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))

        # Query the KDTree with normalized q_p_sample
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)

        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - start_time:.6f} seconds")

        # Step 2: Extract the neighbor points and corresponding secondary modes
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)  # Already normalized
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)

        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - start_time:.6f} seconds")

        # Step 3: Compute pairwise distances between neighbors
        dists_neighbors = squareform(pdist(q_p_neighbors))  # Shape: (k, k)

        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - start_time:.6f} seconds")

        # Step 4: Compute the RBF kernel matrix Psi
        Psi = RBFUtils.multiquadric_rbf(dists_neighbors, epsilon)  # Multiquadric RBF kernel
        # Regularization for numerical stability
        Psi += np.eye(neighbors) * 1e-8

        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - start_time:.6f} seconds")

        # Step 5: Solve for W_neighbors
        W_neighbors = np.linalg.solve(Psi, q_s_neighbors)  # Shape: (k, n_s)

        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - start_time:.6f} seconds")

        # Step 6: Compute Multiquadric RBF values between q_p_sample and its neighbors
        rbf_values = RBFUtils.multiquadric_rbf(dist.flatten(), epsilon)  # Shape: (k,)

        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - start_time:.6f} seconds")

        # Step 7: Compute the Jacobian using vectorized operations
        # Compute the difference matrix D using normalized data
        D = q_p_sample_normalized - q_p_neighbors  # Shape: (k, n_p)

        # Compute the weighted differences for Multiquadric RBF
        weighted_D = (rbf_values[:, np.newaxis]) * D  # Shape: (k, n_p)

        # Compute broadcasting term for Multiquadric kernel derivative
        broadcasting_term = epsilon**2 / np.sqrt(1 + (epsilon * dist.flatten())**2)
        broadcasting_term = broadcasting_term[:, np.newaxis]  # Shape: (k, 1)

        # Compute the Jacobian w.r.t. normalized q_p
        jacobian_normalized = W_neighbors.T @ (weighted_D * broadcasting_term)  # Shape: (n_s, n_p)

        # Adjust the Jacobian to account for the normalization (chain rule)
        scale = scaler.scale_

        # Multiply each column of the Jacobian by scale
        jacobian = jacobian_normalized * scale[np.newaxis, :]  # Broadcasting over columns

        if echo_level >= 1:
            print(f"Time to compute Jacobian: {time.time() - start_time:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def interpolate_with_rbf_nearest_neighbors_dynamic_gaussian(kdtree, q_p_train, q_s_train, q_p_sample_T, epsilon, neighbors, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using nearest neighbors and RBF interpolation dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample_T: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample_T.
        """
        start_time = time.time()

        # Step 1: Find the nearest neighbors in q_p_train
        t0 = time.time()
        # Normalize q_p_sample
        q_p_sample_normalized = scaler.transform(q_p_sample_T.reshape(1, -1))

        if echo_level >= 1:
            print(f"Time to normalize: {time.time() - t0:.6f} seconds")
        
        t0 = time.time()
        # Query the KDTree with normalized q_p_sample
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)

        #dist, idx = kdtree.query(q_p_sample_T.reshape(1, -1), k=neighbors) #Without normalizing
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 2: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 3: Compute pairwise distances between the neighbors
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 4: Compute the RBF matrix for the neighbors
        t0 = time.time()
        Phi_neighbors = RBFUtils.gaussian_rbf(dists_neighbors, epsilon)
        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 5: Solve for the RBF weights (W_neighbors)
        t0 = time.time()
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 6: Compute RBF kernel values between q_p_sample_T and its neighbors
        t0 = time.time()
        rbf_values = RBFUtils.gaussian_rbf(dist.flatten(), epsilon)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 7: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_neighbors
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

    @staticmethod
    def interpolate_with_rbf_nearest_neighbors_dynamic_imq(kdtree, q_p_train, q_s_train, q_p_sample_T, epsilon, neighbors, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using nearest neighbors and Inverse Multiquadric RBF interpolation dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample_T: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.
        - scaler: Scaler for normalization (e.g., StandardScaler or MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample_T.
        """
        start_time = time.time()

        # Step 1: Normalize q_p_sample_T
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample_T.reshape(1, -1))
        if echo_level >= 1:
            print(f"Time to normalize: {time.time() - t0:.6f} seconds")

        # Step 2: Find the nearest neighbors in the normalized space
        t0 = time.time()
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 3: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 4: Compute pairwise distances between the neighbors
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 5: Compute the Inverse Multiquadric RBF matrix for the neighbors
        t0 = time.time()
        Phi_neighbors = RBFUtils.inverse_multiquadric_rbf(dists_neighbors, epsilon)
        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 6: Solve for the RBF weights (W_neighbors)
        t0 = time.time()
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 7: Compute Inverse Multiquadric RBF kernel values between q_p_sample_T and its neighbors
        t0 = time.time()
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist.flatten(), epsilon)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 8: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_neighbors
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

    @staticmethod
    def interpolate_with_rbf_nearest_neighbors_dynamic_linear(kdtree, q_p_train_norm, q_s_train, q_p_sample, epsilon, neighbors, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using nearest neighbors and linear RBF interpolation dynamically,
        accounting for Min-Max normalization.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train_norm: Normalized training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample: The input sample point (unnormalized reduced coordinates, q_p).
        - epsilon: Width parameter for the linear RBF kernel (not used in linear but kept for consistency).
        - neighbors: Number of nearest neighbors to use.
        - scaler: The Min-Max scaler used to normalize the data.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """

        start_time = time.time()

        # Step 1: Normalize q_p_sample
        q_p_sample_norm = scaler.transform(q_p_sample.reshape(1, -1)).reshape(-1)
        
        # Step 2: Find the nearest neighbors in the normalized space
        t0 = time.time()
        dist, idx = kdtree.query(q_p_sample_norm.reshape(1, -1), k=neighbors)
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 3: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train_norm[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 4: Compute pairwise distances between the neighbors in the normalized space
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 5: Compute the RBF matrix for the neighbors (linear kernel)
        t0 = time.time()
        Phi_neighbors = dists_neighbors  # Linear RBF: psi(r) = r
        Phi_neighbors += np.eye(neighbors) * 1e-8  # Regularization for numerical stability
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 6: Solve for the RBF weights (W_neighbors)
        t0 = time.time()
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 7: Compute RBF values between q_p_sample_norm and its neighbors
        t0 = time.time()
        rbf_values = dist.flatten()  # For linear RBF, psi_i = r_i
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 8: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_neighbors
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

    @staticmethod
    def interpolate_with_rbf_nearest_neighbors_dynamic_multiquadric(kdtree, q_p_train, q_s_train, q_p_sample_T, epsilon, neighbors, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using nearest neighbors and Multiquadric RBF interpolation dynamically.

        Parameters:
        - kdtree: KDTree to find nearest neighbors.
        - q_p_train: Training data for principal modes.
        - q_s_train: Training data for secondary modes.
        - q_p_sample_T: The input sample point (reduced coordinates, q_p).
        - epsilon: The width parameter for the RBF kernel.
        - neighbors: Number of nearest neighbors to use.
        - scaler: The StandardScaler used to normalize the data.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample_T.
        """
        start_time = time.time()

        # Step 1: Normalize q_p_sample
        q_p_sample_normalized = scaler.transform(q_p_sample_T.reshape(1, -1))

        # Step 2: Find the nearest neighbors in q_p_train
        t0 = time.time()
        dist, idx = kdtree.query(q_p_sample_normalized, k=neighbors)
        if echo_level >= 1:
            print(f"Time to find nearest neighbors: {time.time() - t0:.6f} seconds")

        # Step 3: Extract the neighbor points and corresponding secondary modes
        t0 = time.time()
        q_p_neighbors = q_p_train[idx].reshape(neighbors, -1)
        q_s_neighbors = q_s_train[idx].reshape(neighbors, -1)
        if echo_level >= 1:
            print(f"Time to extract neighbor data: {time.time() - t0:.6f} seconds")

        # Step 4: Compute pairwise distances between the neighbors
        t0 = time.time()
        dists_neighbors = squareform(pdist(q_p_neighbors))
        if echo_level >= 1:
            print(f"Time to compute pairwise distances: {time.time() - t0:.6f} seconds")

        # Step 5: Compute the Multiquadric RBF matrix for the neighbors
        t0 = time.time()
        Phi_neighbors = RBFUtils.multiquadric_rbf(dists_neighbors, epsilon)
        # Regularization for numerical stability
        Phi_neighbors += np.eye(neighbors) * 1e-8
        if echo_level >= 1:
            print(f"Time to compute RBF matrix and regularize: {time.time() - t0:.6f} seconds")

        # Step 6: Solve for the RBF weights (W_neighbors)
        t0 = time.time()
        W_neighbors = np.linalg.solve(Phi_neighbors, q_s_neighbors)
        if echo_level >= 1:
            print(f"Time to compute RBF weights: {time.time() - t0:.6f} seconds")

        # Step 7: Compute Multiquadric RBF values between q_p_sample_T and its neighbors
        t0 = time.time()
        rbf_values = RBFUtils.multiquadric_rbf(dist.flatten(), epsilon)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 8: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_neighbors
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def compute_rbf_jacobian_global_gaussian(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Gaussian RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # Shape: (num_train,)
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # Shape notes:
        # (q_p_train_norm - x_normalized): (num_train, dim)
        # rbf_values[:, np.newaxis]: (num_train, 1)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = -2 * (epsilon**2) * rbf_values[:, np.newaxis] * (x_normalized - q_p_train_norm)

        # Now compute jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # Shape: (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]

        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_gaussian_no_norm(x, q_p_train, W_global, epsilon, echo_level=0):
        """
        Compute the Jacobian for the Gaussian RBF kernel using a vectorized global approach.

        Parameters:
        - x: np.ndarray, input sample (1 x dim).
        - q_p_train: np.ndarray, primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - x, axis=1)  # Shape: (num_train,)
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # Shape notes:
        # (q_p_train_norm - x_normalized): (num_train, dim)
        # rbf_values[:, np.newaxis]: (num_train, 1)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p = -2 * (epsilon**2) * rbf_values[:, np.newaxis] * (x - q_p_train)

        # Now compute jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian = W_global.T @ Dphi_Dq_p

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_gaussian_no_norm_finite_differences(x, q_p_train, W_global, epsilon, echo_level=0):
        """
        Compute the Jacobian for the Gaussian RBF kernel using a vectorized global approach,
        but via finite differences (no analytical derivative).

        Parameters:
        - x: np.ndarray, input sample (1 x dim).
        - q_p_train: np.ndarray, primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        import time
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train.shape[0]

        # Define a small step for finite differences
        fd_step = 1e-6

        # -------------------------------------------------------------------------
        # Helper function f_func(x_sample) that does the interpolation at x_sample
        # and returns an array of shape (output_dim,).
        # -------------------------------------------------------------------------
        def f_func(x_sample):
            # x_sample is shape (1, dim)
            dist_to_sample = np.linalg.norm(q_p_train - x_sample, axis=1)  # (num_train,)
            rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)    # (num_train,)
            # Interpolate
            f_val = rbf_values @ W_global  # shape: (output_dim,)
            return f_val

        # Evaluate f_func at the current x
        t0 = time.time()
        f_base = f_func(x)
        if echo_level >= 1:
            print(f"Interpolated base value in {time.time() - t0:.6f} seconds")

        # Initialize the Jacobian
        t0 = time.time()
        if echo_level >= 1:
            print("Computing the Jacobian matrix via finite differences...")

        jacobian = np.zeros((output_dim, dim))

        # For each dimension, do a central difference
        for i in range(dim):
            # Create perturbed copies of x
            x_plus  = x.copy()
            x_minus = x.copy()
            x_plus[0, i]  += fd_step
            x_minus[0, i] -= fd_step

            f_plus  = f_func(x_plus)   # shape: (output_dim,)
            f_minus = f_func(x_minus)  # shape: (output_dim,)

            # Central difference
            jacobian[:, i] = (f_plus - f_minus) / (2.0 * fd_step)

        if echo_level >= 1:
            print(f"Computed finite-difference Jacobian in {time.time() - t0:.6f} seconds")

        if echo_level >= 1:
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_global_multiquadric(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = epsilon**2 * (x_normalized - q_p_train_norm) / rbf_values[:, np.newaxis]

        # jacobian_norm: (output_dim, dim) from W_global.T (output_dim x num_train) @ Dphi_Dq_p_norm (num_train x dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust the Jacobian for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_multiquadric_no_norm(x, q_p_train, W_global, epsilon, echo_level=0):
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train.shape[0]

        # Step 1: Compute distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - x, axis=1)  # (num_train,)
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p = epsilon**2 * (x - q_p_train) / rbf_values[:, np.newaxis]

        # jacobian_norm: (output_dim, dim) from W_global.T (output_dim x num_train) @ Dphi_Dq_p_norm (num_train x dim)
        jacobian = W_global.T @ Dphi_Dq_p
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        if echo_level >= 1:
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_global_imq(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Inverse Multiquadric (IMQ) RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute pairwise distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # (x_normalized - q_p_train_norm): (num_train, dim)
        # (rbf_values**3): (num_train,)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p_norm = - (epsilon**2) * (rbf_values**3)[:, np.newaxis] * (x_normalized - q_p_train_norm)

        # Combine with W_global
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # jacobian_norm: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust the Jacobian for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_imq_no_norm(x, q_p_train, W_global, epsilon, echo_level=0):
        """
        Compute the Jacobian for the Inverse Multiquadric (IMQ) RBF kernel using a vectorized global approach.

        Parameters:
        - x: np.ndarray, normalized input sample (1 x dim).
        - q_p_train: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train.shape[0]

        # Step 1: Compute pairwise distances and RBF values
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - x, axis=1)  # (num_train,)
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # φ(r)
        if echo_level >= 1:
            print(f"Computed distances and RBF values in {time.time() - t0:.6f} seconds")

        # Step 2: Compute the Jacobian in a vectorized manner
        t0 = time.time()
        # (x_normalized - q_p_train_norm): (num_train, dim)
        # (rbf_values**3): (num_train,)
        # Dphi_Dq_p_norm: (num_train, dim)
        Dphi_Dq_p = - (epsilon**2) * (rbf_values**3)[:, np.newaxis] * (x - q_p_train)

        # Combine with W_global
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # jacobian_norm: (output_dim, dim)
        jacobian = W_global.T @ Dphi_Dq_p
        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        if echo_level >= 1:
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_imq_no_norm_finite_differences(x, q_p_train, W_global, epsilon, echo_level=0):
        """
        Compute the Jacobian for the Inverse Multiquadric (IMQ) RBF kernel using a vectorized global approach.

        Parameters:
        - x: np.ndarray, normalized input sample (1 x dim).
        - q_p_train: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter.
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train.shape[0]

        # Define a small step for finite differences
        fd_step = 1e-6

        # -------------------------------------------------------------------------
        # Helper function f_func(x_sample) that does the interpolation at x_sample
        # and returns an array of shape (output_dim,).
        # -------------------------------------------------------------------------
        def f_func(x_sample):
            # x_sample is shape (1, dim)
            dist_to_sample = np.linalg.norm(q_p_train - x_sample, axis=1)  # (num_train,)
            rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)    # (num_train,)
            # Interpolate
            f_val = rbf_values @ W_global  # shape: (output_dim,)
            return f_val

        # Evaluate f_func at the current x
        t0 = time.time()
        f_base = f_func(x)
        if echo_level >= 1:
            print(f"Interpolated base value in {time.time() - t0:.6f} seconds")

        # Initialize the Jacobian
        t0 = time.time()
        if echo_level >= 1:
            print("Computing the Jacobian matrix via finite differences...")

        jacobian = np.zeros((output_dim, dim))

        # For each dimension, do a central difference
        for i in range(dim):
            # Create perturbed copies of x
            x_plus  = x.copy()
            x_minus = x.copy()
            x_plus[0, i]  += fd_step
            x_minus[0, i] -= fd_step

            f_plus  = f_func(x_plus)   # shape: (output_dim,)
            f_minus = f_func(x_minus)  # shape: (output_dim,)

            # Central difference
            jacobian[:, i] = (f_plus - f_minus) / (2.0 * fd_step)

        if echo_level >= 1:
            print(f"Computed finite-difference Jacobian in {time.time() - t0:.6f} seconds")

        if echo_level >= 1:
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def compute_rbf_jacobian_global_linear(x_normalized, q_p_train_norm, W_global, epsilon, scaler, echo_level=0):
        """
        Compute the Jacobian for the Linear RBF kernel using a vectorized global approach.

        Parameters:
        - x_normalized: np.ndarray, normalized input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter (unused for linear kernel, but kept for consistency).
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x_normalized.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x_normalized, axis=1)  # (num_train,)
        if echo_level >= 1:
            print(f"Computed distances in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Initialize Dphi_Dq_p_norm with zeros
        Dphi_Dq_p_norm = np.zeros((num_train, dim))

        # Avoid division by zero: only compute for non-zero distances
        nonzero_mask = dist_to_sample > 1e-12
        Dphi_Dq_p_norm[nonzero_mask] = (x_normalized - q_p_train_norm[nonzero_mask]) / dist_to_sample[nonzero_mask, np.newaxis]

        # Combine with W_global
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian_norm = W_global.T @ Dphi_Dq_p_norm

        if echo_level >= 1:
            print(f"Computed Jacobian contributions in {time.time() - t0:.6f} seconds")

        # Step 3: Adjust for Min-Max normalization
        t0 = time.time()
        scale = scaler.scale_  # (dim,)
        jacobian = jacobian_norm * scale[np.newaxis, :]
        if echo_level >= 1:
            print(f"Adjusted Jacobian for Min-Max scaling in {time.time() - t0:.6f} seconds")
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian
    
    @staticmethod
    def compute_rbf_jacobian_global_linear_no_norm(x, q_p_train_norm, W_global, epsilon, echo_level=0):
        """
        Compute the Jacobian for the Linear RBF kernel using a vectorized global approach.

        Parameters:
        - x: np.ndarray, input sample (1 x dim).
        - q_p_train_norm: np.ndarray, normalized primary training coordinates (num_train x dim).
        - W_global: np.ndarray, precomputed weights (num_train x output_dim).
        - epsilon: float, RBF parameter (unused for linear kernel, but kept for consistency).
        - scaler: MinMaxScaler object used for normalization.
        - echo_level: Level of verbosity for timing (default: 0).

        Returns:
        - jacobian: np.ndarray, Jacobian matrix (output_dim x dim).
        """
        start_time = time.time()
        dim = x.shape[1]
        output_dim = W_global.shape[1]
        num_train = q_p_train_norm.shape[0]

        # Step 1: Compute distances
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train_norm - x, axis=1)  # (num_train,)
        if echo_level >= 1:
            print(f"Computed distances in {time.time() - t0:.6f} seconds")

        # Step 2: Compute derivatives in a vectorized manner
        t0 = time.time()
        # Initialize Dphi_Dq_p_norm with zeros
        Dphi_Dq_p_norm = np.zeros((num_train, dim))

        # Avoid division by zero: only compute for non-zero distances
        nonzero_mask = dist_to_sample > 1e-12
        Dphi_Dq_p_norm[nonzero_mask] = (x - q_p_train_norm[nonzero_mask]) / dist_to_sample[nonzero_mask, np.newaxis]

        # Combine with W_global
        # W_global: (num_train, output_dim)
        # W_global.T: (output_dim, num_train)
        # Dphi_Dq_p_norm: (num_train, dim)
        # Result: (output_dim, dim)
        jacobian = W_global.T @ Dphi_Dq_p_norm

        if echo_level >= 1:
            print(f"Total time for Jacobian computation: {time.time() - start_time:.6f} seconds")

        return jacobian

    @staticmethod
    def interpolate_with_rbf_global_gaussian(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Gaussian RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Gaussian RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_gaussian_no_norm(q_p_sample, q_p_train, W_global, epsilon, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Gaussian RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Gaussian RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.gaussian_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

    @staticmethod
    def interpolate_with_rbf_global_multiquadric(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_multiquadric_no_norm(q_p_sample, q_p_train, W_global, epsilon, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_imq(q_p_sample, q_p_train, W_global, epsilon, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Inverse Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Inverse Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_imq_no_norm(q_p_sample, q_p_train, W_global, epsilon, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Inverse Multiquadric RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - epsilon: The width parameter for the RBF kernel.
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Inverse Multiquadric RBF kernel values for the distances
        t0 = time.time()
        rbf_values = RBFUtils.inverse_multiquadric_rbf(dist_to_sample, epsilon)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

    @staticmethod
    def interpolate_with_rbf_global_linear(q_p_sample, q_p_train, W_global, scaler, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Linear RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 1: Normalize the input sample
        t0 = time.time()
        q_p_sample_normalized = scaler.transform(q_p_sample.reshape(1, -1))  # Normalize q_p_sample
        if echo_level >= 1:
            print(f"Time to normalize input: {time.time() - t0:.6f} seconds")

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample_normalized, axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Linear RBF kernel values for the distances
        t0 = time.time()
        rbf_values = dist_to_sample  # Linear kernel: φ(r) = r
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred
    
    @staticmethod
    def interpolate_with_rbf_global_linear_no_norm(q_p_sample, q_p_train, W_global, echo_level=0):
        """
        Interpolate the secondary modes q_s using global Linear RBF interpolation.

        Parameters:
        - q_p_sample: The input sample point (reduced coordinates, q_p).
        - q_p_train: Training data for principal modes.
        - W_global: Precomputed global RBF weights matrix.
        - scaler: Scaler for normalization (e.g., MinMaxScaler).
        - echo_level: Level of verbosity for printing timing information (default: 0).

        Returns:
        - q_s_pred: The predicted secondary modes for the given q_p_sample.
        """
        start_time = time.time()

        # Step 2: Compute pairwise distances between the input sample and all training points
        t0 = time.time()
        dist_to_sample = np.linalg.norm(q_p_train - q_p_sample.reshape(1, -1), axis=1)  # Shape: (num_train,)
        if echo_level >= 1:
            print(f"Time to compute distances: {time.time() - t0:.6f} seconds")

        # Step 3: Compute the Linear RBF kernel values for the distances
        t0 = time.time()
        rbf_values = dist_to_sample  # Linear kernel: φ(r) = r
        if echo_level >= 1:
            print(f"Time to compute RBF values: {time.time() - t0:.6f} seconds")

        # Step 4: Interpolate q_s using the precomputed weights and RBF kernel values
        t0 = time.time()
        q_s_pred = rbf_values @ W_global  # Shape: (num_secondary_modes,)
        if echo_level >= 1:
            print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")
            print(f"Total time: {time.time() - start_time:.6f} seconds")

        return q_s_pred

