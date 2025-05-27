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


def generate_pod_rbf_projection_plot(hdm_snap, prom_snap, mu1, mu2, output_image):
    """Generate and save the POD-RBF PROM projection plot with a title and small margins."""
    num_steps = hdm_snap.shape[1]  # Get the number of timesteps
    inds_to_plot = range(0, num_steps + 1, 100)  # Every 100 time steps

    snaps_to_plot = [hdm_snap, prom_snap]
    labels = ['HDM', 'POD-RBF PROM']
    colors = ['black', 'blue']  # Dark violet for distinction
    linewidths = [3, 3]

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Add a title
    fig.suptitle(f"POD-RBF PROM Projection for $\\mu_1 = {mu1:.2f}$, $\\mu_2 = {mu2:.3f}$", fontsize=14, y=0.98)

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


if __name__ == "__main__":
    # Define test parameter
    mu1, mu2 = 5.19, 0.026  # Example test case

    # Load HDM snapshot
    hdm_file = "../param_snaps/mu1_5.19+mu2_0.026.npy"
    hdm_snap = np.load(hdm_file)

    # Load POD-RBF PROM reconstructed snapshot
    prom_file = "pod_rbf_prom_global_snaps_mu1_5.19_mu2_0.026.npy"
    prom_snap = np.load(prom_file)

    # Generate and save the plot
    output_image = f"mu1_{mu1:.2f}_mu2_{mu2:.3f}_pod_rbf_prom.png"
    generate_pod_rbf_projection_plot(hdm_snap, prom_snap, mu1, mu2, output_image)

