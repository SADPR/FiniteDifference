import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def create_combined_gif(
    npy_files,
    hdm_file,
    labels,
    colors,
    linewidths,
    output_gif,
    num_steps,
    mu1,
    mu2,
    interval=300
):
    """
    Create a combined animation GIF with 2 columns and 2 rows, comparing HDM with each ROM.
    
    Parameters:
        npy_files (list): List of file paths to the `.npy` snapshot files.
        hdm_file (str): Path to the `.npy` file for the HDM.
        labels (list): List of labels for each subplot (one per `.npy` file).
        colors (list): List of colors for each subplot (one per `.npy` file).
        linewidths (list): List of linewidths for each subplot (one per `.npy` file).
        output_gif (str): File path for the output GIF.
        num_steps (int): Number of timesteps to animate.
        mu1 (float): Value of parameter `mu1` to display.
        mu2 (float): Value of parameter `mu2` to display.
        interval (int): Interval in milliseconds between frames.
    """
    # Load the HDM snapshots
    hdm_snaps = np.load(hdm_file)

    # Load the snapshots from the `.npy` files
    snaps_list = [np.load(npy_file) for npy_file in npy_files]

    # Create the figure and subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3, hspace=0.3)

    # Add a subtitle for `mu1` and `mu2`
    fig.suptitle(f"Parameter Values: $\\mu_1 = {mu1:.2f}$, $\\mu_2 = {mu2:.3f}$", fontsize=14, y=0.92)

    # Set the titles for the top row immediately
    for col, label in enumerate(labels):
        axes[0, col].set_title(f"HDM vs {label}")

    # (Optional) Fix y-axis limits for consistent comparison, if desired
    # For example, set them to the same across all frames:
    y_min, y_max = 0, 6.5

    # Prepare dummy lines for the legend so it appears from the start
    # You can import matplotlib.lines as mlines if desired; here we just plot empty lines
    hdm_line = plt.Line2D([], [], color="black", linewidth=2, label="HDM")
    rom_lines = [
        plt.Line2D([], [], color=c, linewidth=lw, label=lbl)
        for c, lw, lbl in zip(colors, linewidths, labels)
    ]
    legend_handles = [hdm_line] + rom_lines

    # Add a single legend at the bottom of the figure
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02)
    )

    def animate_func(frame_idx):
        # Print progress every 10 frames
        if frame_idx % 10 == 0 or frame_idx == num_steps - 1:
            print(f"Processing frame {frame_idx + 1}/{num_steps}...")

        # Clear each subplot for the current frame
        for row in range(2):
            for col in range(2):
                axes[row, col].clear()

        # We reset the titles again, because clearing also erases them
        for col, label in enumerate(labels):
            axes[0, col].set_title(f"HDM vs {label}")

        # For each column in [0, 1], plot HDM vs the corresponding ROM
        for col, (snaps, label, color, lw) in enumerate(zip(snaps_list, labels, colors, linewidths)):
            # Plot the HDM snapshot
            plot_snaps(
                GRID_X, GRID_Y,
                hdm_snaps,
                [frame_idx],  # Only the current frame
                label="HDM",
                fig_ax=(fig, axes[0, col], axes[1, col]),
                color="black",
                linewidth=2
            )
            # Plot the ROM snapshot
            plot_snaps(
                GRID_X, GRID_Y,
                snaps,
                [frame_idx],  # Only the current frame
                label=label,
                fig_ax=(fig, axes[0, col], axes[1, col]),
                color=color,
                linewidth=lw
            )

            # Set consistent y-axis limits if needed
            axes[0, col].set_ylim(y_min, y_max)
            axes[1, col].set_ylim(y_min, y_max)

        return []

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=num_steps,
        interval=interval,
        blit=False,
        repeat=False
    )

    # Save the animation as a GIF
    print(f"Saving animation to '{output_gif}'...")
    anim.save(output_gif, writer="imagemagick", fps=30)
    print(f"Animation saved as '{output_gif}'")


if __name__ == "__main__":
    # Example usage:
    # File paths for the `.npy` snapshot files
    hdm_file = "mu1_4.75+mu2_0.02.npy"  # HDM file
    npy_files = [
        "pod_rbf_hprom_global_snaps_mu1_4.75_mu2_0.020.npy",
        "pod_ann_hprom_snaps_mu1_4.75_mu2_0.020.npy"
    ]

    # Dynamically name the output GIF based on the HDM file
    output_gif = hdm_file.replace(".npy", "_global_vs_ann_comparison_with_params.gif")

    # Labels, colors, and line widths for each simulation
    labels = ["POD-RBF HPROM Global", "POD-ANN HPROM"]
    colors = ["green", "red"]
    linewidths = [2, 2]

    # Parameter values to display
    mu1 = 4.75
    mu2 = 0.02

    # Number of timesteps (assume all `.npy` files have the same number of timesteps)
    sample_snaps = np.load(npy_files[0])
    num_steps = sample_snaps.shape[1]

    # Create the animation
    create_combined_gif(
        npy_files=npy_files,
        hdm_file=hdm_file,
        labels=labels,
        colors=colors,
        linewidths=linewidths,
        output_gif=output_gif,
        num_steps=num_steps,
        mu1=mu1,
        mu2=mu2,
        interval=300
    )
