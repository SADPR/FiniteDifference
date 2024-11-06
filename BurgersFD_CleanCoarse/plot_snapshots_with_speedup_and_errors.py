import os
import numpy as np
import matplotlib.pyplot as plt
from hypernet2D import plot_snaps, make_2D_grid
import re

def print_npz(npz_file):
    """
    Function to print the contents of the .npz file.
    
    npz_file: Path to the .npz file.
    """
    data = np.load(npz_file)
    
    # Print the contents
    for key in data:
        print(f"{key}: {data[key]}")
        
    return data

def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths, grid_x, grid_y, save_path):
    """
    Function to plot snapshots for comparison between different models.
    
    snaps_to_plot: List of snapshot arrays to compare.
    inds_to_plot: List of indices of snapshots to plot.
    labels: Labels for the snapshots.
    colors: Colors for each plot.
    linewidths: Linewidths for each plot.
    grid_x, grid_y: The grid for the plot.
    save_path: Path where the plot will be saved.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, snaps in enumerate(snaps_to_plot):
        plot_snaps(grid_x, grid_y, snaps, inds_to_plot,
                   label=labels[i],
                   fig_ax=(fig, ax1, ax2),
                   color=colors[i],
                   linewidth=linewidths[i])
    
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    # Update the legend placement
    plt.legend(loc='upper right', bbox_to_anchor=(1.013, 1.375), fontsize='xx-small')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved as '{save_path}'")

def plot_npz_with_fom(npz_file, fom_snap_files, prom_snap_files, rnm_snap_files, pod_rbf_snap_files, grid_x, grid_y, output_folder):
    """
    Function to load the snapshots, plot them, and calculate speedup.
    
    npz_file: Path to the .npz file.
    fom_snap_files: List of FOM snapshot files for each parameter combination.
    prom_snap_files: List of PROM snapshot files for each parameter combination.
    rnm_snap_files: List of RNM snapshot files for each parameter combination.
    pod_rbf_snap_files: List of POD-RBF snapshot files for each parameter combination.
    grid_x, grid_y: Grid data for the plots.
    output_folder: Folder where the plots will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Print the .npz file contents and retrieve data
    data = print_npz(npz_file)

    # Loop through each combination of parameter pairs and plot the snapshots
    for i, (fom_file, prom_file, rnm_file, pod_rbf_file) in enumerate(zip(fom_snap_files, prom_snap_files, rnm_snap_files, pod_rbf_snap_files)):
        # Extract mu1 and mu2 from the filename using regex
        mu1_match = re.search(r"mu1_([\d.]+)", fom_file)
        mu2_match = re.search(r"mu2_([\d.]+)", fom_file)
        
        if mu1_match and mu2_match:
            mu1 = mu1_match.group(1)
            mu2 = mu2_match.group(1)
        else:
            raise ValueError(f"Could not extract mu1 and mu2 from filename: {fom_file}")
        
        # Load the snapshots from the FOM and ROMs
        fom_snaps = np.load(fom_file)
        prom_snaps = np.load(prom_file)
        rnm_snaps = np.load(rnm_file)
        pod_rbf_snaps = np.load(pod_rbf_file)

        # Define indices for plotting
        inds_to_plot = range(0, fom_snaps.shape[1], 100)
        labels = ['HDM', 'PROM', 'POD-ANN', 'POD-RBF']
        colors = ['black', 'blue', 'green', 'red']
        linewidths = [2, 1, 1, 1]

        # Plot and save each comparison (PROM, POD-ANN, POD-RBF) for each parameter combination
        # Save paths for each plot
        save_prom_path = os.path.join(output_folder, f'prom_mu1_{mu1}_mu2_{mu2}png')
        save_rnm_path = os.path.join(output_folder, f'rnm_mu1_{mu1}_mu2_{mu2}png')
        save_pod_rbf_path = os.path.join(output_folder, f'pod_rbf_mu1_{mu1}_mu2_{mu2}png')

        # Plot and save each model's comparison with FOM
        compare_snaps([fom_snaps, prom_snaps], inds_to_plot, labels[:2], colors[:2], linewidths[:2], grid_x, grid_y, save_prom_path)
        compare_snaps([fom_snaps, rnm_snaps], inds_to_plot, [labels[0], labels[2]], [colors[0], colors[2]], [linewidths[0], linewidths[2]], grid_x, grid_y, save_rnm_path)
        compare_snaps([fom_snaps, pod_rbf_snaps], inds_to_plot, [labels[0], labels[3]], [colors[0], colors[3]], [linewidths[0], linewidths[3]], grid_x, grid_y, save_pod_rbf_path)

    # Calculate and print speedup with respect to HDM (FOM)
    fom_time = data['fom_times']  # Assuming the FOM time is saved as 'fom_times' in the npz
    prom_time = data['prom_times']
    rnm_time = data['rnm_times']
    pod_rbf_time = data['pod_rbf_times']

    # Calculate speedups and ensure we handle scalar or array values
    prom_speedup_scalar = fom_time / prom_time if np.isscalar(fom_time) else np.mean(fom_time / prom_time)
    rnm_speedup_scalar = fom_time / rnm_time if np.isscalar(fom_time) else np.mean(fom_time / rnm_time)
    pod_rbf_speedup_scalar = fom_time / pod_rbf_time if np.isscalar(fom_time) else np.mean(fom_time / pod_rbf_time)

    # Print speedups
    print(f"PROM speedup: {prom_speedup_scalar:.2f}x")
    print(f"POD-ANN speedup: {rnm_speedup_scalar:.2f}x")
    print(f"POD-RBF speedup: {pod_rbf_speedup_scalar:.2f}x")

if __name__ == "__main__":
    # Define file paths and grid data
    npz_file = 'rom_results.npz'
    output_folder = 'rom_plots'

    num_cells_x, num_cells_y = 250, 250  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid

    # Define the mu1 and mu2 parameters
    mu1_values = [5.19, 4.56, 4.75]
    mu2_values = [0.026, 0.019, 0.02]

    # Dynamically generate snapshot file paths based on mu1 and mu2 values
    fom_snap_files = [f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    prom_snap_files = [f'rom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    rnm_snap_files = [f'rnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    pod_rbf_snap_files = [f'pod_rbf_prom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]

    # Run the plotting and speedup calculation
    plot_npz_with_fom(npz_file, fom_snap_files, prom_snap_files, rnm_snap_files, pod_rbf_snap_files, grid_x, grid_y, output_folder)


