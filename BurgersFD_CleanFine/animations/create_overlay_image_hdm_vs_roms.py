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

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)


def create_overlay_image(npy_files, hdm_file, labels, colors, linewidths, output_image, num_steps, mu1, mu2):
    """
    Create a static overlay plot comparing HDM with each ROM for a given set of timesteps.

    Parameters:
        npy_files (list): List of file paths to the `.npy` snapshot files.
        hdm_file (str): Path to the `.npy` file for the HDM.
        labels (list): List of labels for each subplot (one per `.npy` file).
        colors (list): List of colors for each subplot (one per `.npy` file).
        linewidths (list): List of linewidths for each subplot (one per `.npy` file).
        output_image (str): File path for the output PNG image.
        num_steps (int): Number of timesteps (to define the `inds_to_plot`).
        mu1 (float): Value of parameter `mu1` to display.
        mu2 (float): Value of parameter `mu2` to display.
    """
    # Load the HDM snapshots
    hdm_snaps = np.load(hdm_file)

    # Load the snapshots from the `.npy` files
    snaps_list = [np.load(npy_file) for npy_file in npy_files]

    # Timesteps to plot
    inds_to_plot = range(0, num_steps + 1, 100)  # Every 100 timesteps

    # Create the figure and subplots (2 rows, 3 columns for three datasets)
    fig, axes = plt.subplots(2, 3, figsize=(21, 8))  # Adjust the figure size for three columns

    # Remove outer margins, but keep inner spacing
    fig.subplots_adjust(left=0.03, right=0.97, top=0.85, bottom=0.13, wspace=0.3, hspace=0.3)

    # Add a subtitle for `mu1` and `mu2`
    fig.suptitle(f"Parameter Values: $\\mu_1 = {mu1:.2f}$, $\\mu_2 = {mu2:.3f}$", fontsize=14, y=0.94)

    # Collect legend handles
    legend_handles = []

    for col, (snaps, label, color, linewidth) in enumerate(zip(snaps_list, labels, colors, linewidths)):
        # Plot HDM vs. ROM for each column
        hdm_line, = axes[0, col].plot([], [], color="black", linewidth=2, label="HDM")  # Dummy line for HDM
        rom_line, = axes[0, col].plot([], [], color=color, linewidth=linewidth, label=label)  # Dummy line for ROM

        # Plot the actual data
        plot_snaps(
            GRID_X, GRID_Y, hdm_snaps, inds_to_plot,
            label="HDM",
            fig_ax=(fig, axes[0, col], axes[1, col]),
            color="black",
            linewidth=2
        )
        plot_snaps(
            GRID_X, GRID_Y, snaps, inds_to_plot,
            label=label,
            fig_ax=(fig, axes[0, col], axes[1, col]),
            color=color,
            linewidth=linewidth
        )

        # Set titles for the first row only
        axes[0, col].set_title(f"HDM vs {label}")

        # Add to legend handles only once
        if col == 0:
            legend_handles.append(hdm_line)
        legend_handles.append(rom_line)

    # Add a single legend at the center of the entire figure
    fig.legend(handles=legend_handles, loc="center", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.03))

    # Save the figure
    print(f"Saving overlay image to '{output_image}'...")
    plt.savefig(output_image, dpi=300)  # No need for bbox_inches here since outer margins are already reduced
    print(f"Overlay image saved as '{output_image}'")
    plt.show()


if __name__ == "__main__":
    # File paths for the `.npy` snapshot files
    hdm_file = "mu1_4.25+mu2_0.0225.npy"  # HDM file
    npy_files = [
        "pod_ann_hprom_snaps_mu1_4.25_mu2_0.022.npy",
        "pod_rbf_hprom_global_snaps_mu1_4.25_mu2_0.022.npy",
        "pod_gp_hprom_snaps_mu1_4.25_mu2_0.022.npy"  # Added third file
    ]

    # Dynamically name the output image based on the HDM file
    output_image = hdm_file.replace(".npy", "_overlay_with_params.png")

    # Labels, colors, and line widths for each simulation
    labels = ["POD-ANN HPROM", "POD-RBF HPROM", "POD-GP HPROM"]  # Added third label
    colors = ["red", "blue", "green"]  # Added third color
    linewidths = [2, 2, 2]  # Line width for all three plots

    # Parameter values to display
    mu1 = 4.25
    mu2 = 0.0225

    # Number of timesteps (assume all `.npy` files have the same number of timesteps)
    sample_snaps = np.load(npy_files[0])
    num_steps = sample_snaps.shape[1]

    # Create the overlay image
    create_overlay_image(npy_files, hdm_file, labels, colors, linewidths, output_image, num_steps, mu1, mu2)
