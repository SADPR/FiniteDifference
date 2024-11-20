import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from hypernet2D import plot_snaps, make_2D_grid
import re


def create_animation(snaps_to_plot, inds_to_plot, labels, colors, linewidths, grid_x, grid_y, output_file, fps=30):
    """
    Create and save an animation comparing snapshots over time.

    snaps_to_plot: List of snapshot arrays to compare.
    inds_to_plot: Indices of snapshots to plot (frames).
    labels: Labels for the snapshots.
    colors: Colors for each plot.
    linewidths: Linewidths for each plot.
    grid_x, grid_y: The grid for the plots.
    output_file: Path to save the animation (GIF format).
    fps: Frames per second for the animation.
    """
    interval = 1000 / fps  # Milliseconds per frame
    fig, (ax1, ax2) = plt.subplots(2, 1)

    # Initialize plots
    plots = [
        plot_snaps(grid_x, grid_y, np.zeros_like(snaps_to_plot[0]), inds_to_plot[:1],
                   label=label, fig_ax=(fig, ax1, ax2), color=color, linewidth=lw)
        for snaps, label, color, lw in zip(snaps_to_plot, labels, colors, linewidths)
    ]

    def init():
        """Initialize the plot."""
        for ax in [ax1, ax2]:
            ax.clear()
            ax.grid()
            ax.set_ylim(0, 6.5)  # Fixed Y-axis range
        return plots

    def update(frame):
        """Update the plot for each frame."""
        ax1.clear()
        ax2.clear()
        ax1.set_ylim(0, 6.5)  # Fixed Y-axis range
        ax2.set_ylim(0, 6.5)  # Fixed Y-axis range

        for i, snaps in enumerate(snaps_to_plot):
            plot_snaps(grid_x, grid_y, snaps, [frame],
                       label=labels[i], fig_ax=(fig, ax1, ax2), color=colors[i], linewidth=linewidths[i])

        ax1.legend(loc='upper right', fontsize='xx-small')
        return plots

    ani = FuncAnimation(fig, update, frames=inds_to_plot, init_func=init, interval=interval, blit=False)

    # Save the animation as GIF
    ani.save(output_file, writer="pillow", dpi=300, fps=fps)
    print(f"Animation saved as '{output_file}'")


if __name__ == "__main__":
    # Define file paths and grid data
    output_folder = 'hrom_animations'

    num_cells_x, num_cells_y = 750, 750  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the mu1 and mu2 parameters
    mu1_values = [5.19, 4.56, 4.75]
    mu2_values = [0.026, 0.019, 0.02]

    # Dynamically generate snapshot file paths based on mu1 and mu2 values
    fom_snap_files = [f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    hprom_snap_files = [f'dd_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    hrnm_snap_files = [f'dd_hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    pod_rbf_hprom_snap_files = [f'dd_pod_rbf_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]

    # Load the snapshots
    for i, (fom_file, hprom_file, hrnm_file, pod_rbf_hprom_file) in enumerate(zip(fom_snap_files, hprom_snap_files, hrnm_snap_files, pod_rbf_hprom_snap_files)):
        # Extract mu1 and mu2 from the filename
        mu1_match = re.search(r"mu1_([\d.]+)", fom_file)
        mu2_match = re.search(r"mu2_([\d.]+)", fom_file)
        mu1 = mu1_match.group(1)
        mu2 = mu2_match.group(1)

        # Load the snapshots
        fom_snaps = np.load(fom_file)
        hprom_snaps = np.load(hprom_file)
        hrnm_snaps = np.load(hrnm_file)
        pod_rbf_hprom_snaps = np.load(pod_rbf_hprom_file)

        # Define indices for animation (snapshots from 0 to 500)
        inds_to_plot = range(0, 500)  # Include all 500 time steps

        # Create separate animations for each ROM type
        create_animation(
            snaps_to_plot=[fom_snaps, hprom_snaps],
            inds_to_plot=inds_to_plot,
            labels=['HDM', 'POD HPROM'],
            colors=['black', 'blue'],
            linewidths=[2, 2],
            grid_x=grid_x,
            grid_y=grid_y,
            output_file=os.path.join(output_folder, f"animation_pod_hprom_mu1_{mu1}_mu2_{mu2}.gif"),
            fps=30  # Smooth playback with 30 frames per second
        )

        create_animation(
            snaps_to_plot=[fom_snaps, hrnm_snaps],
            inds_to_plot=inds_to_plot,
            labels=['HDM', 'POD-ANN HPROM'],
            colors=['black', 'green'],
            linewidths=[2, 2],
            grid_x=grid_x,
            grid_y=grid_y,
            output_file=os.path.join(output_folder, f"animation_pod_ann_hprom_mu1_{mu1}_mu2_{mu2}.gif"),
            fps=30
        )

        create_animation(
            snaps_to_plot=[fom_snaps, pod_rbf_hprom_snaps],
            inds_to_plot=inds_to_plot,
            labels=['HDM', 'POD-RBF HPROM'],
            colors=['black', 'red'],
            linewidths=[2, 2],
            grid_x=grid_x,
            grid_y=grid_y,
            output_file=os.path.join(output_folder, f"animation_pod_rbf_hprom_mu1_{mu1}_mu2_{mu2}.gif"),
            fps=30
        )
