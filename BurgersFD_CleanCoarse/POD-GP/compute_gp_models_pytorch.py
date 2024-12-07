# compute_gp_models.py

import os
import numpy as np
import time
import pickle
import logging
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps, make_2D_grid

def get_snapshot_params():
    """
    Generate a list of parameter vectors [mu1, mu2] within specified ranges.
    """
    MU1_RANGE = 4.25, 5.5
    MU2_RANGE = 0.015, 0.03
    SAMPLES_PER_MU = 3

    MU1_LOW, MU1_HIGH = MU1_RANGE
    MU2_LOW, MU2_HIGH = MU2_RANGE
    mu1_samples = np.linspace(MU1_LOW, MU1_HIGH, SAMPLES_PER_MU)
    mu2_samples = np.linspace(MU2_LOW, MU2_HIGH, SAMPLES_PER_MU)
    mu_samples = []
    for mu1 in mu1_samples:
        for mu2 in mu2_samples:
            mu_samples.append([mu1, mu2])
    return mu_samples

def perform_pod(snaps, num_modes=150, method='rsvd', random_state=None):
    """
    Perform Proper Orthogonal Decomposition (POD) using SVD or Randomized SVD.
    """
    if method == 'rsvd':
        print("Performing Randomized SVD for POD...")
        start_time = time.time()
        from sklearn.utils.extmath import randomized_svd
        U, sigma, Vh = randomized_svd(snaps, n_components=num_modes, random_state=random_state)
        elapsed_time = time.time() - start_time
        print(f"Randomized SVD completed in {elapsed_time:.2f} seconds.")
    elif method == 'svd':
        print("Performing standard SVD for POD...")
        start_time = time.time()
        U, s, Vh = np.linalg.svd(snaps, full_matrices=False)
        U = U[:, :num_modes]
        sigma = s[:num_modes]
        elapsed_time = time.time() - start_time
        print(f"Standard SVD completed in {elapsed_time:.2f} seconds.")
    else:
        raise ValueError("Invalid POD method. Choose 'svd' or 'rsvd'.")

    basis = U
    return basis, sigma

def main():
    # Configure logging to capture missing snapshots
    logging.basicConfig(filename='missing_snapshots.log', level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # Define simulation parameters
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    # Initial condition (replace with actual initial condition as needed)
    w0 = np.ones((num_cells_x * num_cells_y * 2,))  # Example initial condition

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Ensure the snapshot folder exists
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        print("Please add the required snapshot files before running the script again.")
        return  # Exit since no snapshots are available

    # Generate all parameter samples
    mu_samples = get_snapshot_params()
    print(f"Total parameter samples: {len(mu_samples)}")

    # Attempt to load the shape of the first snapshot to determine snapshot dimensions
    try:
        first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        snapshot_shape = first_snap.shape
        print(f"Shape of each snapshot: {snapshot_shape}")
    except FileNotFoundError as e:
        print(f"Error loading the first snapshot: {e}")
        print("Ensure that at least one snapshot exists to determine the snapshot dimensions.")
        logging.error(f"Error loading the first snapshot: {e}")
        return  # Exit since snapshot dimensions are unknown

    snap_count = len(mu_samples)  # Total number of parameter combinations
    total_snaps = snapshot_shape[1] * snap_count  # Total number of snapshots (time steps * parameter combinations)
    print(f"Total number of snapshots to aggregate: {total_snaps}")

    # Pre-allocate memory for all snapshots
    snaps = np.zeros((snapshot_shape[0], total_snaps))

    # Collect snapshots into the pre-allocated array
    col_offset = 0
    successful_mu = []
    missing_mu = []

    for idx, mu in enumerate(mu_samples):
        try:
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
            snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu  # Insert directly
            col_offset += snap_mu.shape[1]  # Update column offset for the next parameter set
            successful_mu.append(mu)
            print(f"Loaded snapshot {idx + 1}/{snap_count} for mu1={mu[0]}, mu2={mu[1]}")
        except FileNotFoundError as e:
            print(e)
            missing_mu.append(mu)
            logging.info(f"Missing snapshots for mu1={mu[0]}, mu2={mu[1]}")

    # Trim the pre-allocated array in case some snapshots are missing
    if missing_mu:
        loaded_snaps = col_offset
        snaps = snaps[:, :loaded_snaps]
    else:
        loaded_snaps = total_snaps  # All snapshots loaded

    print(f"Successfully loaded {loaded_snaps} snapshots out of {total_snaps}.")

    if missing_mu:
        print("Missing parameter sets have been logged in 'missing_snapshots.log'.")

    if snaps.size == 0:
        print("No snapshots were loaded. Exiting the workflow.")
        return

    print(f"Combined snapshot matrix shape: {snaps.shape}")

    # Define whether to compute the basis or load a precomputed one
    compute_basis = False  # Set to False to load a precomputed basis

    if not compute_basis:
        # Load a precomputed basis
        basis_path = 'basis.npy'
        if os.path.exists(basis_path):
            basis = np.load(basis_path, allow_pickle=True)
            print(f"Loaded precomputed basis from {basis_path}.")
        else:
            print(f"Basis file '{basis_path}' not found. Please compute the basis first.")
            return
    else:
        # Define POD parameters
        pod_method = 'rsvd'  # Choose between 'svd' or 'rsvd'
        num_modes = 150  # Total number of POD modes to retain
        random_state = 42  # For reproducibility (only used if pod_method='rsvd')

        # Perform POD to compute the Reduced Order Basis (ROB)
        basis, sigma = perform_pod(snaps, num_modes=num_modes, method=pod_method, random_state=random_state)
        print(f"Computed POD basis with method '{pod_method}' and {num_modes} modes.")

        # Save the computed basis and singular values for future use
        np.save('basis.npy', basis)
        np.save('sigma.npy', sigma)
        print("Saved computed basis to 'basis.npy' and singular values to 'sigma.npy'.")

    # Define how many primary modes to use
    primary_modes = 10
    total_modes = 150  # Ensure total_modes >= primary_modes

    # Project the snapshots onto the POD basis
    print("Projecting snapshots onto the POD basis...")
    projection_start_time = time.time()
    q = basis.T @ snaps  # Project snapshots onto the POD basis
    q_p = q[:primary_modes, :]  # Primary mode projections
    q_s = q[primary_modes:total_modes, :]  # Secondary mode projections
    print(f"Projection took {time.time() - projection_start_time:.2f} seconds.")

    # Normalize q_p using Min-Max normalization and save the scaler
    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T  # Note the transpose operations
    print("Normalization of q_p complete.")

    # Save the scaler for q_p
    modes_dir = "modes"
    if not os.path.exists(modes_dir):
        os.makedirs(modes_dir)
        print(f"Created modes directory: {modes_dir}")

    with open(os.path.join(modes_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Scale q_s using StandardScaler and save the scaler
    print("Scaling q_s data using StandardScaler...")
    y_scaler = StandardScaler()
    q_s_scaled = y_scaler.fit_transform(q_s.T).T  # Shape: (num_secondary_modes, num_samples)
    print("Scaling of q_s complete.")

    # Save the scaler for q_s
    with open(os.path.join(modes_dir, 'y_scaler.pkl'), 'wb') as f:
        pickle.dump(y_scaler, f)
    print("y_scaler saved successfully.")

    # Save the normalized q_p and q_s for future use
    np.save(os.path.join(modes_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(modes_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(modes_dir, 'q.npy'), q)
    np.save(os.path.join(modes_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(modes_dir, 'q_s_scaled.npy'), q_s_scaled)
    print("Primary and secondary modes, as well as projected data (q, q_p_normalized, q_s_scaled), saved successfully.")

    # Prepare training data
    X_train = q_p_normalized.T  # Shape: (num_samples, num_primary_modes)
    y_train = q_s_scaled.T      # Shape: (num_samples, num_secondary_modes)

    # Convert training data to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # Create DataLoader for batch processing
    batch_size = 256  # Reduced batch size to save memory
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Force the code to use CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Define the GP model
    class MultitaskGPModel(ApproximateGP):
        def __init__(self, inducing_points, num_tasks):
            self.num_tasks = num_tasks
            batch_shape = torch.Size([num_tasks])

            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0), batch_shape=batch_shape
            )
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy, num_tasks=num_tasks
            )

            super(MultitaskGPModel, self).__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = ScaleKernel(
                RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape
            )

        def forward(self, x):
            # x shape: [batch_size, input_dim]
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialize inducing points
    num_inducing = 100  # Adjust based on memory constraints
    inducing_points = X_train_tensor[:num_inducing, :].to(device)

    num_tasks = y_train_tensor.shape[1]
    print(f"Number of tasks (num_tasks): {num_tasks}")

    model = MultitaskGPModel(inducing_points=inducing_points, num_tasks=num_tasks).to(device)
    likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

    # Set up the optimizer
    num_epochs = 5  # Adjust based on desired training time
    learning_rate = 0.005
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    # Set up the loss function
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_train_tensor.size(0))

    # Training loop
    model.train()
    likelihood.train()

    print("Starting training...")
    training_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)

            print(f"output.mean.shape: {output.mean.shape}")  # Should be [num_tasks, batch_size]
            print(f"output.covariance_matrix.shape: {output.covariance_matrix.shape}")
            print(f"y_batch.shape: {y_batch.shape}")  # Should be [batch_size, num_tasks]

            # Transpose y_batch to match output.mean
            y_batch_t = y_batch.transpose(0, 1)  # Shape: [num_tasks, batch_size]

            loss = -mll(output, y_batch_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    print(f"Training completed in {time.time() - training_start_time:.2f} seconds.")

    # Save the trained model
    model_filename = os.path.join(modes_dir, 'gpytorch_model_state.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'scaler': scaler,
        'y_scaler': y_scaler,
        'device': device.type,
    }, model_filename)
    print(f"Model saved successfully in {model_filename}.")

    print("Processing complete.")

if __name__ == '__main__':
    main()