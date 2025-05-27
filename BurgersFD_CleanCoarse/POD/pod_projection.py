import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary modules
from hypernet2D import load_or_compute_snaps, plot_snaps
from config import GRID_X, GRID_Y  # Use pre-defined grid from config

# Matplotlib settings for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)


def reconstruct_snapshot_with_pod(snapshot, U, num_modes, print_times=False):
    """Reconstruct a snapshot using standard POD."""
    start_time = time.time()
    U_modes = U[:, :num_modes]
    q_pod = U_modes.T @ snapshot
    reconstructed_pod = U_modes @ q_pod
    if print_times:
        print(f"POD reconstruction took: {time.time() - start_time:.6f} seconds")
    return reconstructed_pod


def generate_pod_projection_plot(hdm_snap, pod_reconstructed, mu1, mu2, num_modes, output_image):
    """Generate and save the POD projection plot with a title and small margins."""
    num_steps = hdm_snap.shape[1]  # Get the number of timesteps
    inds_to_plot = range(0, num_steps + 1, 100)  # Every 100 time steps

    snaps_to_plot = [hdm_snap, pod_reconstructed]
    labels = ['HDM', f'POD ({num_modes} modes): Projection']
    colors = ['black', '#9400D3']
    linewidths = [3, 3]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Add a title
    fig.suptitle(f"POD Projection for $\\mu_1 = {mu1:.2f}$, $\\mu_2 = {mu2:.3f}$", fontsize=14, y=0.98)

    for snap, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths):
        plot_snaps(
            GRID_X, GRID_Y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax1, ax2),
            color=color,
            linewidth=lw
        )

    # Add legend to both subplots
    ax2.legend(loc="center right", fontsize=20, frameon=True)

    # Adjust layout with small margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0.2)

    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Comparison plot saved successfully to {output_image}")




if __name__ == '__main__':
    # Define test parameter
    mu1, mu2 = 4.56, 0.019  # Example test case
    num_modes = 150

    # Flag to control saving of the `.npy` file
    SAVE_NPY = False  # Change to False to disable saving

    # Load POD basis
    U_full = np.load('basis.npy', allow_pickle=True)

    # Define simulation parameters
    dt, num_steps = 0.05, 500
    snap_folder = "../param_snaps"

    # Ensure snapshot folder exists
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
        print(f"Created snapshot directory: {snap_folder}")
        exit(1)

    # Load the specific snapshot for the target parameter pair
    try:
        hdm_snap = load_or_compute_snaps([mu1, mu2], GRID_X, GRID_Y, np.ones((GRID_X.size * 2,)), dt, num_steps, snap_folder=snap_folder)
        print(f"Loaded HDM snapshot for mu1={mu1}, mu2={mu2}")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Perform POD reconstruction
    pod_reconstructed = reconstruct_snapshot_with_pod(hdm_snap, U_full, num_modes, print_times=True)

    # Save reconstructed snapshot if the flag is enabled
    if SAVE_NPY:
        pod_file_path = f"mu1_{mu1:.2f}_mu2_{mu2:.3f}_pod_projection.npy"
        np.save(pod_file_path, pod_reconstructed)
        print(f"Standard POD reconstructed snapshot saved successfully to {pod_file_path}")

    # Compute relative error
    pod_error = np.linalg.norm(hdm_snap - pod_reconstructed) / np.linalg.norm(hdm_snap)
    print(f"Standard POD Reconstruction error: {pod_error:.6e}")

    # Generate and save the plot
    output_image = f"mu1_{mu1:.2f}_mu2_{mu2:.3f}_pod_projection.png"
    generate_pod_projection_plot(hdm_snap, pod_reconstructed, mu1, mu2, num_modes, output_image)
