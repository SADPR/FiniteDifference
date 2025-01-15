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


def create_slice_animation(npy_files, labels, colors, linewidths, output_gif, num_steps, slice_axis, interval=300):
    """
    Create an animation for a specific slice (x-mid or y-mid) plotted for all snapshots in one figure.

    Parameters:
        npy_files (list): List of file paths to the `.npy` snapshot files.
        labels (list): List of labels for each line (one per `.npy` file).
        colors (list): List of colors for each line (one per `.npy` file).
        linewidths (list): List of linewidths for each line (one per `.npy` file).
        output_gif (str): File path for the output GIF.
        num_steps (int): Number of timesteps to animate.
        slice_axis (str): 'x' for x-midpoint slice or 'y' for y-midpoint slice.
        interval (int): Interval in milliseconds between frames.
    """
    # Load the snapshots from the `.npy` files
    snaps_list = [np.load(npy_file) for npy_file in npy_files]

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Spatial grid
    x = (GRID_X[1:] + GRID_X[:-1]) / 2  # Midpoints in x
    y = (GRID_Y[1:] + GRID_Y[:-1]) / 2  # Midpoints in y
    mid_x = int(x.size / 2)  # Midpoint index along x-axis
    mid_y = int(y.size / 2)  # Midpoint index along y-axis

    # Set up axis labels and limits
    if slice_axis == 'x':
        ax.set_xlabel("$x$")
        ax.set_ylabel("$u_x(x, y={:.1f})$".format(y[mid_y]))
        ax.set_xlim(0, x[-1])
    elif slice_axis == 'y':
        ax.set_xlabel("$y$")
        ax.set_ylabel("$u_x(x={:.1f}, y)$".format(x[mid_x]))
        ax.set_xlim(0, y[-1])
    ax.set_ylim(0, 6.5)  # Adjust based on solution range
    ax.legend()
    ax.grid()

    # Initialize the lines for each simulation
    lines = [ax.plot([], [], label=labels[i], color=colors[i], linewidth=linewidths[i])[0] for i in range(len(snaps_list))]

    def animate_func(frame_idx):
        # Print progress every 10 frames
        if frame_idx % 10 == 0 or frame_idx == num_steps - 1:
            print(f"Processing frame {frame_idx + 1}/{num_steps}...")

        for line, snaps in zip(lines, snaps_list):
            # Extract the current frame's 2D snapshot
            snap = snaps[:(y.size * x.size), frame_idx].reshape(y.size, x.size)

            # Update lines based on the slice axis
            if slice_axis == 'x':
                line.set_data(x, snap[mid_y, :])  # Horizontal slice
            elif slice_axis == 'y':
                line.set_data(y, snap[:, mid_x])  # Vertical slice

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

    # Number of timesteps (assume all `.npy` files have the same number of timesteps)
    sample_snaps = np.load(npy_files[0])
    num_steps = sample_snaps.shape[1]

    # Output GIF files
    output_gif_x = "mu1_4.75+mu2_0.02_x_midpoint.gif"
    output_gif_y = "mu1_4.75+mu2_0.02_y_midpoint.gif"

    # Create animations for both x-midpoint and y-midpoint slices
    create_slice_animation(npy_files, labels, colors, linewidths, output_gif_x, num_steps, slice_axis='x')
    create_slice_animation(npy_files, labels, colors, linewidths, output_gif_y, num_steps, slice_axis='y')
