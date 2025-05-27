import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import GRID_X, GRID_Y

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)


def create_single_plot_animation_with_slices(npy_files, labels, colors, linewidths, output_gif, num_steps, interval=300):
    """
    Create an animation with specific slices (midpoints along x and y) plotted for all snapshots in one figure.
    
    Parameters:
        npy_files (list): List of file paths to the `.npy` snapshot files.
        labels (list): List of labels for each line (one per `.npy` file).
        colors (list): List of colors for each line (one per `.npy` file).
        linewidths (list): List of linewidths for each line (one per `.npy` file).
        output_gif (str): File path for the output GIF.
        num_steps (int): Number of timesteps to animate.
        interval (int): Interval in milliseconds between frames.
    """
    # Load the snapshots from the `.npy` files
    snaps_list = [np.load(npy_file) for npy_file in npy_files]

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.3)  # Adjust spacing between subplots

    # Spatial grid
    x = (GRID_X[1:] + GRID_X[:-1]) / 2  # Midpoints in x
    y = (GRID_Y[1:] + GRID_Y[:-1]) / 2  # Midpoints in y
    mid_x = int(x.size / 2)  # Midpoint index along x-axis
    mid_y = int(y.size / 2)  # Midpoint index along y-axis

    # Initialize the lines for each simulation
    lines_x = [ax1.plot([], [], label=f"{label} (x-mid)", color=colors[i], linewidth=linewidths[i])[0] for i, label in enumerate(labels)]
    lines_y = [ax2.plot([], [], label=f"{label} (y-mid)", color=colors[i], linewidth=linewidths[i])[0] for i, label in enumerate(labels)]

    # Set axis labels and limits
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u_x(x, y={:.1f})$".format(y[mid_y]))
    ax1.set_xlim(0, x[-1])
    ax1.set_ylim(0, 6.5)  # Adjust based on solution range
    ax1.legend()
    ax1.grid()

    ax2.set_xlabel("$y$")
    ax2.set_ylabel("$u_x(x={:.1f}, y)$".format(x[mid_x]))
    ax2.set_xlim(0, y[-1])
    ax2.set_ylim(0, 6.5)  # Adjust based on solution range
    ax2.legend()
    ax2.grid()

    def animate_func(frame_idx):
        # Print progress every 10 frames
        if frame_idx % 1 == 0 or frame_idx == num_steps - 1:
            print(f"Processing frame {frame_idx + 1}/{num_steps}...")

        for line_x, line_y, snaps in zip(lines_x, lines_y, snaps_list):
            # Extract the current frame's 2D snapshot
            snap = snaps[:(y.size * x.size), frame_idx].reshape(y.size, x.size)

            # Update lines for x-mid and y-mid slices
            line_x.set_data(x, snap[mid_y, :])  # Horizontal slice
            line_y.set_data(y, snap[:, mid_x])  # Vertical slice

    # Create the animation
    anim = animation.FuncAnimation(
        fig, animate_func,
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
    # File paths for the `.npy` snapshot files
    npy_files = [
        "mu1_4.75+mu2_0.02.npy",  # HDM
        "pod_rbf_hprom_nearest_snaps_mu1_4.75_mu2_0.020.npy",
        "pod_rbf_hprom_global_snaps_mu1_4.75_mu2_0.020.npy",
        "pod_ann_hprom_snaps_mu1_4.75_mu2_0.020.npy"
    ]

    # Labels, colors, and line widths for each simulation
    labels = ["HDM", "POD-RBF HPROM Nearest", "POD-RBF HPROM Global", "POD-ANN HPROM"]
    colors = ["black", "blue", "green", "red"]
    linewidths = [2, 2, 2, 2]

    # Output GIF file
    output_gif = "combined_all_in_one_with_slices_animation.gif"

    # Number of timesteps (assume all `.npy` files have the same number of timesteps)
    sample_snaps = np.load(npy_files[0])
    num_steps = sample_snaps.shape[1]

    # Create the animation
    create_single_plot_animation_with_slices(npy_files, labels, colors, linewidths, output_gif, num_steps)
