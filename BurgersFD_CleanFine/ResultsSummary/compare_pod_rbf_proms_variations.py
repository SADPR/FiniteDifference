import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hypernet2D import plot_snaps
from config import GRID_X, GRID_Y

# Matplotlib settings for LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)


def compute_relative_error(hdm_snap, prom_snap):
    """Compute the relative error between the HDM and PROM/HPROM solutions."""
    return np.linalg.norm(hdm_snap - prom_snap) / np.linalg.norm(hdm_snap) * 100


def generate_pod_rbf_comparison_plot(hdm_snap, snaps_list, labels, colors, linewidths, mu1, mu2, output_image):
    """Generate and save the POD-RBF PROM vs. HPROM projection plot with a title and small margins."""
    num_steps = hdm_snap.shape[1]  # Get the number of timesteps
    inds_to_plot = range(0, num_steps + 1, 100)  # Every 100 time steps

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Add a title
    fig.suptitle(f"POD-RBF Projection for $\\mu_1 = {mu1:.2f}$, $\\mu_2 = {mu2:.3f}$", fontsize=14, y=0.98)

    # Plot HDM first
    plot_snaps(
        GRID_X, GRID_Y, hdm_snap, inds_to_plot,
        label="HDM",
        fig_ax=(fig, ax1, ax2),
        color="black",
        linewidth=3
    )

    # Compute errors and plot each PROM/HPROM variation
    for snap, label, color, lw in zip(snaps_list, labels, colors, linewidths):
        relative_error = compute_relative_error(hdm_snap, snap)
        print(f"Relative error for {label}: {relative_error:.2f}%")

        plot_snaps(
            GRID_X, GRID_Y, snap, inds_to_plot,
            label=f"{label}",
            fig_ax=(fig, ax1, ax2),
            color=color,
            linewidth=lw
        )

    # Add legend to both subplots
    ax2.legend(loc="center right", fontsize=16, frameon=True)

    # Adjust layout with small margins
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0.2)

    # Save the plot
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Comparison plot saved successfully to {output_image}")


if __name__ == "__main__":
    # Define test parameter
    mu1, mu2 = 4.75, 0.020  # Example test case

    # Load HDM snapshot
    hdm_file = "../param_snaps/mu1_4.75+mu2_0.02.npy"
    hdm_snap = np.load(hdm_file)

    # Load different PROM/HPROM reconstructions
    files = [
        "pod_rbf_prom_global_snaps_mu1_4.75_mu2_0.020.npy",
        "pod_rbf_hprom_global_snaps_mu1_4.75_mu2_0.020_joshua.npy",
        "pod_rbf_hprom_global_snaps_mu1_4.75_mu2_0.020.npy"
    ]

    labels = [
        "POD-RBF PROM",
        "POD-RBF HPROM (Joshua's ECSW mesh)",
        "POD-RBF HPROM (New ECSW mesh)"
    ]
    
    colors = ["blue", "red", "green"]  # Blue for PROM, red for Joshua, green for new ECSW mesh
    linewidths = [3, 3, 3]  # Keep all at same thickness

    # Load snapshots
    snaps_list = [np.load(file) for file in files]

    # Generate and save the plot
    output_image = f"mu1_{mu1:.2f}_mu2_{mu2:.3f}_pod_rbf_hprom_comparison.png"
    generate_pod_rbf_comparison_plot(hdm_snap, snaps_list, labels, colors, linewidths, mu1, mu2, output_image)

