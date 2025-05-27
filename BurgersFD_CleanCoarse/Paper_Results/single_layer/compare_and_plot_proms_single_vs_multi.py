import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import GRID_X, GRID_Y

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]
})
plt.rc('font', size=13)

def plot_snaps(grid_x, grid_y, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid',
               label=None, fig_ax=None, alpha=1.0):
    if fig_ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, ax1, ax2 = fig_ax

    x = (grid_x[1:] + grid_x[:-1]) / 2
    y = (grid_y[1:] + grid_y[:-1]) / 2
    mid_x = int(x.size / 2)
    mid_y = int(y.size / 2)
    is_first_line = True
    for ind in snaps_to_plot:
        label2 = label if is_first_line else None
        is_first_line = False
        snap = snaps[:(y.size * x.size), ind].reshape(y.size, x.size)
        ax1.plot(x, snap[mid_y, :], color=color, linestyle=linestyle, linewidth=linewidth, label=label2, alpha=alpha)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u_x(x,y={:0.1f})$'.format(y[mid_y]))
        ax1.grid()
        ax2.plot(y, snap[:, mid_x], color=color, linestyle=linestyle, linewidth=linewidth, label=label2, alpha=alpha)
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u_x(x={:0.1f},y)$'.format(x[mid_x]))
        ax2.grid()
    return fig, ax1, ax2

def compute_relative_error(hdm_snap, prom_snap):
    return (np.linalg.norm(hdm_snap - prom_snap) / np.linalg.norm(hdm_snap)) * 100.0

def generate_comparison_plot(hdm_snap, snap_dict, mu1, mu2, output_image):
    num_steps = hdm_snap.shape[1]
    inds_to_plot = range(0, num_steps + 1, 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
    fig.suptitle(
        rf"$\mu_1 = {mu1:.2f}, \mu_2 = {mu2:.3f}$",
        fontsize=13,
        y=0.98
    )

    labels = list(snap_dict.keys())
    colors = ["teal", "crimson"]

    # Plot HDM reference
    plot_snaps(GRID_X, GRID_Y, hdm_snap, inds_to_plot,
               label="HDM",
               fig_ax=(fig, ax1, ax2),
               color="black",
               linewidth=3,
               alpha=1.0)

    # Plot single-layer and muti-layer ANN ROMs
    for i, label in enumerate(labels):
        snap = snap_dict[label]
        rel_err = compute_relative_error(hdm_snap, snap)
        print(f"Relative error for {label} = {rel_err:.2f}%")
        plot_snaps(
            GRID_X, GRID_Y, snap, inds_to_plot,
            label=label,
            fig_ax=(fig, ax1, ax2),
            color=colors[i],
            linewidth=2,
            alpha=0.9
        )

    ax1.legend(loc="center left", fontsize=10, frameon=True)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0.3)
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved comparison plot: {output_image}")
    plt.close(fig)

if __name__ == "__main__":
    mu1, mu2 = 4.56, 0.019

    # File paths
    hdm_file = f"../hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    ann_single_file = f"pod_ann_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"
    ann_multi_file  = f"../pod_ann_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy"

    # Load data
    hdm_snap = np.load(hdm_file)
    ann_single_snap = np.load(ann_single_file)
    ann_multi_snap  = np.load(ann_multi_file)

    # Dictionary: Only two models
    model_snaps = {
        "HPROM-ANN (single-layer)": ann_single_snap,
        "HPROM-ANN (muti-layer)": ann_multi_snap,
    }

    # Output image
    out_png = f"ann_single_vs_multilayer_mu1_{mu1:.2f}_mu2_{mu2:.3f}.png"
    generate_comparison_plot(hdm_snap, model_snaps, mu1, mu2, out_png)

    print("Done!")

