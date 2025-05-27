# interpolate_gp.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ensure a non-interactive backend is used
import matplotlib.pyplot as plt
import pickle  # Changed from joblib to pickle
import sys
import time

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import required functions
from hypernet2D import load_or_compute_snaps, make_2D_grid, plot_snaps

def reconstruct_snapshot_with_gp(snapshot, U_p, U_s, scaler, gp_model, print_times=False):
    """
    Reconstruct snapshots using a single, multi-dimensional Gaussian Process
    model that was trained on *unscaled* q_s.

    Parameters:
    - snapshot : Full high-dimensional snapshot(s) to reconstruct.
    - U_p, U_s : POD basis matrices for primary and secondary modes.
    - scaler   : MinMaxScaler for q_p only.
    - gp_model : The fitted GP model (unscaled q_s).
    - print_times : (bool) Print timing information if True.
    """

    total_start_time = time.time()

    # Step 1: Project snapshot onto primary coordinates
    step1_start = time.time()
    q = U_p.T @ snapshot
    q_p = q[:U_p.shape[1], :]
    step1_time = time.time() - step1_start

    # Step 2: Normalize q_p
    step2_start = time.time()
    q_p_normalized = scaler.transform(q_p.T).T  # Shape: (num_primary_modes, num_time_steps)
    step2_time = time.time() - step2_start

    # Step 3: Predict q_s using the GP model
    step3_start = time.time()
    num_time_steps = q_p_normalized.shape[1]
    q_s_pred = []

    if print_times:
        print("Predicting q_s for each time step using GP model...")

    for i in range(num_time_steps):
        q_p_sample = q_p_normalized[:, i].reshape(1, -1)  # (1, num_primary_modes)
        # gp_model.predict(...) -> shape (1, num_secondary_modes)
        q_s_sample_unscaled = gp_model.predict(q_p_sample)
        q_s_pred.append(q_s_sample_unscaled.flatten())

    # Convert from list to array of shape (num_secondary_modes, num_time_steps)
    q_s_pred = np.array(q_s_pred).T
    step3_time = time.time() - step3_start

    # Step 4: Reconstruct the full snapshots
    step4_start = time.time()
    reconstructed_snapshots_gp = []
    for i in range(num_time_steps):
        reconstructed_snapshot_gp = U_p @ q_p[:, i] + U_s @ q_s_pred[:, i]
        reconstructed_snapshots_gp.append(reconstructed_snapshot_gp)

    reconstructed_snapshots_gp = np.array(reconstructed_snapshots_gp).T  # (num_dofs, num_time_steps)
    step4_time = time.time() - step4_start

    # Print timing if requested
    if print_times:
        total_time = time.time() - total_start_time
        print("[reconstruct_snapshot_with_gp] timing breakdown:")
        print(f"  Step 1 (project snapshot): {step1_time:.6f} s")
        print(f"  Step 2 (normalize q_p): {step2_time:.6f} s")
        print(f"  Step 3 (GP predictions): {step3_time:.6f} s")
        print(f"  Step 4 (reconstruct): {step4_time:.6f} s")
        print(f"  Total reconstruction time: {total_time:.6f} s")

    return reconstructed_snapshots_gp


if __name__ == '__main__':
    # Define the parameter pair you want to reconstruct and compare
    target_mu = [4.56, 0.019]  # Example: mu1=5.19, mu2=0.026

    # Define simulation parameters
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)

    # Initial condition
    w0 = np.ones((num_cells_x * num_cells_y * 2,))  # Example initial condition

    # Define the folder where snapshots are stored
    snap_folder = "../param_snaps"

    # Ensure the snapshot folder exists
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        print("Please add the required snapshot files before running the script again.")
        exit(1)  # Exit since no snapshots are available

    # Load the specific snapshot for the target parameter pair
    try:
        hdm_snap = load_or_compute_snaps(target_mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        print(f"Loaded HDM snapshot for mu1={target_mu[0]}, mu2={target_mu[1]}")
    except FileNotFoundError as e:
        print(e)
        exit(1)  # Exit since the target snapshot is not available

    # Load the GP model (trained on unscaled q_s)
    model_dir = "pod_gp_model"  # Changed from "modes"
    try:
        gp_models_filename = os.path.join(model_dir, 'gp_model.pkl')
        with open(gp_models_filename, 'rb') as f:
            gp_model = pickle.load(f)
        print("Single multi-output GP model (unscaled q_s) loaded successfully.")
    except FileNotFoundError:
        print(f"File '{gp_models_filename}' not found.")
        exit(1)

    # Load U_p and U_s
    try:
        U_p = np.load(os.path.join(model_dir, 'U_p.npy'))
        U_s = np.load(os.path.join(model_dir, 'U_s.npy'))
        U_full = np.hstack((U_p, U_s))  # Full U matrix with all modes
        print("POD basis matrices (U_p and U_s) loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Load the saved scaler (Min-Max scaler for q_p)
    try:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("Min-Max scaler for q_p loaded successfully.")
    except FileNotFoundError:
        print("Scaler file 'scaler.pkl' not found.")
        exit(1)

    # Additional parameters
    num_modes = U_p.shape[1] + U_s.shape[1]
    compare_pod = True  # Set to True to include Standard POD reconstruction
    print_times = True

    # Reconstruct the snapshot using GP interpolation (q_s unscaled)
    gp_reconstructed = reconstruct_snapshot_with_gp(
        hdm_snap, U_p, U_s, scaler, gp_model, print_times
    )

    # Reconstruct the snapshot using standard POD with all modes
    def reconstruct_snapshot_with_pod(snapshot, U, num_modes, print_times=False):
        """Reconstruct a snapshot using standard POD."""
        start_time = time.time()
        U_modes = U[:, :num_modes]
        q_pod = U_modes.T @ snapshot
        reconstructed_pod = U_modes @ q_pod
        if print_times:
            elapsed = time.time() - start_time
            print(f"POD reconstruction took: {elapsed:.6f} seconds")
        return reconstructed_pod

    pod_reconstructed = None
    if compare_pod:
        pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, U_full, num_modes, print_times)

    # Save the reconstructed data
    results_dir = "FOM_vs_GP_Reconstruction_Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    gp_file_path = os.path.join(
        results_dir, f"reconstructed_snapshot_gp_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
    )
    np.save(gp_file_path, gp_reconstructed)
    print(f"GP reconstructed snapshot saved successfully to {gp_file_path}")

    if compare_pod and pod_reconstructed is not None:
        pod_file_path = os.path.join(
            results_dir, f"reconstructed_snapshot_pod_mu1_{target_mu[0]}_mu2_{target_mu[1]}.npy"
        )
        np.save(pod_file_path, pod_reconstructed)
        print(f"Standard POD reconstructed snapshot saved successfully to {pod_file_path}")

    # Calculate and compare reconstruction errors
    gp_error = np.linalg.norm(hdm_snap - gp_reconstructed) / np.linalg.norm(hdm_snap)
    print(f"GP Reconstruction error: {gp_error:.6e}")

    if compare_pod and pod_reconstructed is not None:
        pod_error = np.linalg.norm(hdm_snap - pod_reconstructed) / np.linalg.norm(hdm_snap)
        print(f"Standard POD Reconstruction error: {pod_error:.6e}")

    # Define indices to plot (e.g., specific time steps)
    inds_to_plot = range(0, num_steps + 1, 100)  # Example: every 100 time steps

    # Prepare snapshots to plot
    snaps_to_plot = [hdm_snap, gp_reconstructed]
    labels = ['HDM', 'GP']
    colors = ['black', 'red']
    linewidths = [2, 2]

    if compare_pod and pod_reconstructed is not None:
        snaps_to_plot.append(pod_reconstructed)
        labels.append('Standard POD')
        colors.append('blue')
        linewidths.append(1)

    # Plot the comparison using subplots for x and y slices
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(
            grid_x, grid_y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax1, ax2),
            color=color,
            linewidth=lw
        )

    # Print relative errors
    relative_error_gp = 100 * gp_error
    print('Relative error (GP): {:3.2f}%'.format(relative_error_gp))

    if compare_pod and pod_reconstructed is not None:
        relative_error_pod = 100 * pod_error
        print('Relative error (Standard POD): {:3.2f}%'.format(relative_error_pod))

    # Finalize and save the plot
    plt.tight_layout()
    plt.grid(True)
    plt.legend(loc='upper right')
    plot_filename = f"plot_mu1_{target_mu[0]:.2f}_mu2_{target_mu[1]}.png"
    plt.savefig(os.path.join(results_dir, plot_filename), dpi=300)
    print(f"Comparison plot saved successfully to {os.path.join(results_dir, plot_filename)}")

    # Optionally, display the plot
    # plt.show()
