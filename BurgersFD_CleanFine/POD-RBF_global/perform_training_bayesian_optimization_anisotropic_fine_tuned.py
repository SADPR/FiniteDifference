import os
import numpy as np
import time
import pickle
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

def gaussian_rbf(r, epsilon):
    """Gaussian RBF kernel function."""
    return np.exp(-(epsilon * r) ** 2)

def inverse_multiquadric_rbf(r, epsilon):
    """Inverse Multiquadric RBF kernel function."""
    return 1.0 / np.sqrt(1 + (epsilon * r) ** 2)

def matern_kernel(r, epsilon):
    """Matérn kernel function with nu=3/2.
    
    k(r) = (1 + sqrt(3)*epsilon*r) * exp(-sqrt(3)*epsilon*r)
    """
    sqrt3 = np.sqrt(3)
    return (1 + sqrt3 * epsilon * r) * np.exp(-sqrt3 * epsilon * r)

rbf_kernels = {
    'gaussian': gaussian_rbf,
    'imq': inverse_multiquadric_rbf,
    'matern': matern_kernel
}

def anisotropic_distance_matrix_cross(X_val, X_train, alpha):
    """
    Returns the distance from each X_val[i] to each X_train[j],
    i.e. shape (len(X_val), len(X_train)).

    X_val: (n_val, n_dims)
    X_train: (n_train, n_dims)
    alpha: (n_dims,) scale factors
    """
    n_val, n_dims = X_val.shape
    n_train = X_train.shape[0]
    dist_mat = np.zeros((n_val, n_train), dtype=np.float64)

    # Scale each dimension of X_val and X_train
    scaled_val = X_val * alpha  # shape (n_val, n_dims)
    scaled_train = X_train * alpha  # shape (n_train, n_dims)

    for i in range(n_val):
        # shape (n_train, n_dims)
        diff = scaled_train - scaled_val[i]  # broadcast along rows
        dist_mat[i, :] = np.sqrt(np.sum(diff**2, axis=1))

    return dist_mat


def main():
    # Load a precomputed basis
    basis_path = 'basis.npy'
    basis = np.load(basis_path, allow_pickle=True)
    print(f"Loaded precomputed basis from {basis_path}.")
    
    # Define how many primary modes to use
    primary_modes = 10
    total_modes = 150  # Ensure total_modes >= primary_modes

    q_p = np.load('q_p.npy')
    q_s = np.load('q_s.npy')
    
    # Normalize q_p using Min-Max normalization and save the scaler
    print("Normalizing q_p data using Min-Max normalization...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    q_p_normalized = scaler.fit_transform(q_p.T).T  # Note the transpose operations
    print("Normalization complete.")

    # Save the scaler for future use
    model_dir = "pod_rbf_global_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved successfully.")

    # Save the normalized q_p and q_s for future use
    np.save(os.path.join(model_dir, 'U_p.npy'), basis[:, :primary_modes])
    np.save(os.path.join(model_dir, 'U_s.npy'), basis[:, primary_modes:total_modes])
    np.save(os.path.join(model_dir, 'q_p_normalized.npy'), q_p_normalized)
    np.save(os.path.join(model_dir, 'q_s.npy'), q_s)
    print("Primary and secondary modes, as well as projected data (q_p_normalized, q_s), saved successfully.")

    # Prepare training data (each row is a snapshot)
    q_p_train = q_p_normalized.T  # Shape: (n_snapshots, num_primary_modes)
    q_s_train = q_s.T             # Shape: (n_snapshots, num_secondary_modes)

    # Create a train–validation split (90% training, 10% validation)
    #X_train, X_val, y_train, y_val = train_test_split(
    #    q_p_train, q_s_train, test_size=0.2, random_state=42
    #)

    # -------------------------------------------------------------------
    # Added section: Project test snapshots for the new parameter sets.
    # Test parameters: [mu1, mu2] pairs: [4.75, 0.02], [4.56, 0.019], [5.19, 0.026]
    test_params = [
        [4.56, 0.019]
    ]
    print("Projecting test snapshots onto the POD basis...")

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Pre-allocate test snapshots matrix (each test snapshot has the same shape as training snapshots)
    n_test = len(test_params)
    total_test_snaps = 501 * n_test
    snaps_test = np.zeros((basis.shape[0], total_test_snaps))
    col_offset = 0
    for mu in test_params:
        test_snap = load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)
        snaps_test[:, col_offset:col_offset + test_snap.shape[1]] = test_snap
        col_offset += test_snap.shape[1]
        print(f"Loaded test snapshot for mu1={mu[0]}, mu2={mu[1]}")

    # Project the aggregated test snapshots onto the POD basis
    q_test = basis.T @ snaps_test
    # Split test projection into primary and secondary modes
    q_p_test_original = q_test[:primary_modes, :]
    q_s_test = q_test[primary_modes:total_modes, :]
    q_p_test = scaler.transform(q_p_test_original.T).T
    q_p_test = q_p_test.T  # Now shape: (n_test_snapshots, num_primary_modes)
    q_s_test = q_s_test.T  # Now shape: (n_test_snapshots, num_secondary_modes)

    # Define the search space for Bayesian optimization
    # Instead of one log_epsilon, you do:
    space = [
        Real(np.log(1e-2), np.log(5), name='log_alpha1'),
        Real(np.log(1e-2), np.log(5), name='log_alpha2'),
        Real(np.log(1e-2), np.log(5), name='log_alpha3'),
        Real(np.log(1e-2), np.log(5), name='log_alpha4'),
        Real(np.log(1e-2), np.log(5), name='log_alpha5'),
        Real(np.log(1e-2), np.log(5), name='log_alpha6'),
        Real(np.log(1e-2), np.log(5), name='log_alpha7'),
        Real(np.log(1e-2), np.log(5), name='log_alpha8'),
        Real(np.log(1e-2), np.log(5), name='log_alpha9'),
        Real(np.log(1e-2), np.log(5), name='log_alpha10'),
        Categorical(list(rbf_kernels.keys()), name='kernel_name'),
        Real(np.log(1e-10), np.log(1e-3), name='log_lambda_val')
    ]

    print("Optimizing alpha, kernel and lambda using a train–validation split and bayesian optimization...")
    # Define the objective function
    @use_named_args(space)
    def objective(log_alpha1, log_alpha2, log_alpha3, log_alpha4, log_alpha5, log_alpha6, log_alpha7, log_alpha8, log_alpha9, log_alpha10, kernel_name, log_lambda_val):
        alpha = np.array([
        np.exp(log_alpha1),
        np.exp(log_alpha2),
        np.exp(log_alpha3),
        np.exp(log_alpha4),
        np.exp(log_alpha5),
        np.exp(log_alpha6),
        np.exp(log_alpha7),
        np.exp(log_alpha8),
        np.exp(log_alpha9),
        np.exp(log_alpha10),
        ])
        kernel_func = rbf_kernels[kernel_name]
        lambda_val = np.exp(log_lambda_val)

        dists_train_aniso = anisotropic_distance_matrix_cross(q_p_train, q_p_train, alpha)
        
        Phi_train = kernel_func(dists_train_aniso, 1.0)
        Phi_train += np.eye(Phi_train.shape[0]) * lambda_val  # Regularization

        try:
            W_cv = np.linalg.solve(Phi_train, q_s_train)
        except np.linalg.LinAlgError:
            print(f"LinAlgError at alpha={alpha}, kernel={kernel_name}, lambda={lambda_val} ...")
            return np.inf

        # Compute distance matrix for the validation split (relative to training)
        dists_val_aniso = anisotropic_distance_matrix_cross(q_p_test, q_p_train, alpha)
        
        Phi_val = kernel_func(dists_val_aniso, 1.0)
        y_val_pred = Phi_val @ W_cv

        # Compute validation errors
        error = mean_squared_error(q_s_test, y_val_pred)
        rpe = (np.linalg.norm(q_s_test - y_val_pred, 'fro') /
                    np.linalg.norm(q_s_test, 'fro')) * 100

        print(f"Alpha={alpha}, Kernel={kernel_name}, Lambda={lambda_val:.2e}, "
                f"Val MSE={error:.5e}, Val Relative Error={rpe:.2f}%")


        return rpe

    # Start timing
    start_time = time.time()

    # Perform Bayesian optimization
    print("Optimizing alpha and kernel using Bayesian optimization...")
    # After we do:
    result = gp_minimize(
        objective,
        space,
        acq_func='EI',
        n_calls=100,
        random_state=42,
        verbose=True,
        n_random_starts=9
    )

    # End timing
    end_time = time.time()

    # Extract results
    # result.x is a list of length 12: [log_alpha1..log_alpha10, kernel_name, log_lambda_val]
    best_log_alphas = result.x[:10]  # the first 10 entries
    best_kernel_name = result.x[10]  # the 11th
    best_log_lambda_val = result.x[11]  # the 12th

    # Convert to numeric
    best_alpha = np.exp(best_log_alphas)  # shape (10,)
    best_lambda_val = np.exp(best_log_lambda_val)

    print(f"Best alpha found: {best_alpha}")
    print(f"Best kernel found: {best_kernel_name}")
    print(f"Best lambda found: {best_lambda_val:.3e}")
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")

    # =================
    #  RECOMPUTE W ON *ALL* TRAINING DATA
    # =================

    kernel_func = rbf_kernels[best_kernel_name]

    # 1) Build anisotropic distances among all training points q_p_train
    dists_train_full = anisotropic_distance_matrix_cross(q_p_train, q_p_train, best_alpha)
    # e.g. shape (N, N)

    # 2) Evaluate the kernel with "epsilon=1.0" because we've done dimensionwise scaling in alpha
    Phi_train_full = kernel_func(dists_train_full, 1.0)

    # 3) Add Tikhonov diagonal
    Phi_train_full += np.eye(len(q_p_train)) * best_lambda_val

    # 4) Solve for W
    try:
        W_final = np.linalg.solve(Phi_train_full, q_s_train)
    except np.linalg.LinAlgError:
        print("LinAlgError when solving for the final weight matrix with best alpha & lambda. Exiting.")
        return

    # 5) Save the final model to disk
    training_data_filename = os.path.join(model_dir, 'global_weights.pkl')
    with open(training_data_filename, 'wb') as f:
        pickle.dump({
            'W': W_final,                # shape (N, n_s)
            'q_p_train': q_p_train,      # shape (N, 10)
            'q_s_train': q_s_train,      # shape (N, n_s)
            'alpha': best_alpha,         # shape (10,)
            'kernel_name': best_kernel_name,
            'lambda_val': best_lambda_val
        }, f)

    print(f"Global weight matrix and data saved in {training_data_filename}.")
    print("Processing complete.")


if __name__ == '__main__':
    main()
