import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#from hypernet2D import plot_snaps
from config import GRID_X, GRID_Y

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)

def compute_relative_error(hdm_snap, prom_snap):
    """
    Compute the relative error between the HDM and a PROM/HPROM solution
    in the usual L2 norm sense.
    """
    return (np.linalg.norm(hdm_snap - prom_snap) / np.linalg.norm(hdm_snap)) * 100.0

def plot_snaps(grid_x, grid_y, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid',
               label=None, fig_ax=None, xlim=None, ylim=None, slice_axis='x'):

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    x = (grid_x[1:] + grid_x[:-1]) / 2
    y = (grid_y[1:] + grid_y[:-1]) / 2
    mid_x = int(x.size / 2)
    mid_y = int(y.size / 2)

    is_first_line = True  # To set labels only for the first plot
    for ind in snaps_to_plot:
        label2 = label if is_first_line else None  # Label only for first plot
        is_first_line = False
        
        snap = snaps[:(y.size * x.size), ind].reshape(y.size, x.size)

        if slice_axis == 'x':
            ax.plot(x, snap[mid_y, :], color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
            ax.set_xlabel('$x$')
            ax.set_ylabel(f"$u_x(x, y={y[mid_y]:.1f})$")
        else:
            ax.plot(y, snap[:, mid_x], color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
            ax.set_xlabel('$y$')
            ax.set_ylabel(f"$u_x(x={x[mid_x]:.1f}, y)$")

    ax.grid()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return fig, ax

def generate_comparison_plot(
    hdm_snap,
    snap_dict,
    mu1,
    mu2,
    output_image
):
    """
    Generate a (4,2) subplot: 
      - Left column → Y cut planes
      - Right column → X cut planes
    """

    num_steps = hdm_snap.shape[1]  # Number of timesteps in the data
    inds_to_plot = range(0, num_steps + 1, 100)  # Sampled timesteps

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(
        rf"Comparison for $\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$",
        fontsize=16,
        y=0.95
    )

    colors = ["darkgoldenrod", "red", "green", "blue"]
    linestyles = ["solid", "solid", "solid", "solid"]

    models = list(snap_dict.keys())

    # Plot each model separately in its own subplot (left = Y cut, right = X cut)
    for i, (label, snap) in enumerate(snap_dict.items()):
        rel_err = compute_relative_error(hdm_snap, snap)
        print(f"Relative error for {label} = {rel_err:.2f}%")

        ax_y = axes[i, 0]  # Left column (Y cut)
        ax_x = axes[i, 1]  # Right column (X cut)

        plot_snaps(
            GRID_X, GRID_Y, hdm_snap, inds_to_plot,
            label="HDM",
            fig_ax=(fig, ax_x),
            color="black",
            linewidth=3,
            slice_axis='y'  # Ensure Y-slice in the left column
        )

        plot_snaps(
            GRID_X, GRID_Y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax_x),
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            slice_axis='y'  # Ensure Y-slice in the left column
        )

        plot_snaps(
            GRID_X, GRID_Y, hdm_snap, inds_to_plot,
            label="HDM",
            fig_ax=(fig, ax_y),
            color="black",
            linewidth=3,
            slice_axis='x'  # Ensure X-slice in the right column
        )

        plot_snaps(
            GRID_X, GRID_Y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax_y),
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            slice_axis='x'  # Ensure X-slice in the right column
        )

        ax_y.set_title(f"{label} ($y = 50.2$)")
        ax_x.set_title(f"{label} ($x = 50.2$)")

    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved comparison plot: {output_image}")
    plt.close(fig)

if __name__ == "__main__":
    # Example usage for one test point. Adjust paths and filenames to match your naming.
    mu1, mu2 = 5.19, 0.026

    # Load HDM data
    hdm_file = f"hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    hdm_snap = np.load(hdm_file)

    # Load each HPROM variant:
    pod_hprom_file = f"pod_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    pod_ann_file   = f"pod_ann_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    pod_gp_file    = f"pod_gp_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    pod_rbf_file   = f"pod_rbf_hprom_global_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"

    pod_hprom_snap = np.load(pod_hprom_file)
    pod_ann_snap   = np.load(pod_ann_file)
    pod_gp_snap    = np.load(pod_gp_file)
    pod_rbf_snap   = np.load(pod_rbf_file)

    # Dictionary of models
    model_snaps = {
        "HPROM": pod_hprom_snap,
        "HPROM-ANN": pod_ann_snap,
        "HPROM-GP": pod_gp_snap,
        "HPROM-RBF": pod_rbf_snap,
    }

    # Generate plot
    out_png = f"comparison_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png"
    generate_comparison_plot(hdm_snap, model_snaps, mu1, mu2, out_png)

    print("Done!")
