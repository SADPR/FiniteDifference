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

def plot_npz_with_fom(npz_file, fom_snap_files, hprom_snap_files, hrnm_snap_files, pod_rbf_hprom_snap_files, grid_x, grid_y, output_folder):
    """
    Function to load the snapshots, plot them, and calculate speedup.
    
    npz_file: Path to the .npz file.
    fom_snap_files: List of FOM snapshot files for each parameter combination.
    hprom_snap_files: List of HPROM snapshot files for each parameter combination.
    hrnm_snap_files: List of HRNM snapshot files for each parameter combination.
    pod_rbf_hprom_snap_files: List of POD-RBF HPROM snapshot files for each parameter combination.
    grid_x, grid_y: Grid data for the plots.
    output_folder: Folder where the plots will be saved.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Print the .npz file contents and retrieve data
    data = print_npz(npz_file)

    # Loop through each combination of parameter pairs and plot the snapshots
    for i, (fom_file, hprom_file, hrnm_file, pod_rbf_hprom_file) in enumerate(zip(fom_snap_files, hprom_snap_files, hrnm_snap_files, pod_rbf_hprom_snap_files)):
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
        hprom_snaps = np.load(hprom_file)
        hrnm_snaps = np.load(hrnm_file)
        pod_rbf_hprom_snaps = np.load(pod_rbf_hprom_file)

        # Define indices for plotting
        inds_to_plot = range(0, fom_snaps.shape[1], 100)
        labels = ['HDM', 'POD HPROM', 'POD-ANN HPROM', 'POD-RBF HPROM']
        colors = ['black', 'blue', 'green', 'red']
        linewidths = [2, 1, 1, 1]

        # Plot and save each model's comparison with FOM
        # Save paths for each plot
        save_hprom_path = os.path.join(output_folder, f'hprom_mu1_{mu1}_mu2_{mu2}_M2png')
        save_hrnm_path = os.path.join(output_folder, f'hrnm_mu1_{mu1}_mu2_{mu2}_M2png')
        save_pod_rbf_hprom_path = os.path.join(output_folder, f'pod_rbf_hprom_mu1_{mu1}_mu2_{mu2}_M2png')

        compare_snaps([fom_snaps, hprom_snaps], inds_to_plot, labels[:2], colors[:2], linewidths[:2], grid_x, grid_y, save_hprom_path)
        compare_snaps([fom_snaps, hrnm_snaps], inds_to_plot, [labels[0], labels[2]], [colors[0], colors[2]], [linewidths[0], linewidths[2]], grid_x, grid_y, save_hrnm_path)
        compare_snaps([fom_snaps, pod_rbf_hprom_snaps], inds_to_plot, [labels[0], labels[3]], [colors[0], colors[3]], [linewidths[0], linewidths[3]], grid_x, grid_y, save_pod_rbf_hprom_path)

    # Calculate and print speedup with respect to HDM (FOM)
    fom_time = data['fom_times']  # Assuming the FOM time is saved as 'fom_times' in the npz
    hprom_time = data['hprom_times']
    hrnm_time = data['hrnm_times']
    pod_rbf_hprom_time = data['pod_rbf_hprom_times']

    # Convert lists to NumPy arrays for element-wise operations
    fom_time = np.array(fom_time)
    hprom_time = np.array(hprom_time)
    hrnm_time = np.array(hrnm_time)
    pod_rbf_hprom_time = np.array(pod_rbf_hprom_time)

    # Calculate speedups for each method
    hprom_speedup = fom_time / hprom_time
    hrnm_speedup = fom_time / hrnm_time
    pod_rbf_hprom_speedup = fom_time / pod_rbf_hprom_time

    # Calculate average and max speedups
    hprom_speedup_avg = np.mean(hprom_speedup)
    hprom_speedup_max = np.max(hprom_speedup)

    hrnm_speedup_avg = np.mean(hrnm_speedup)
    hrnm_speedup_max = np.max(hrnm_speedup)

    pod_rbf_hprom_speedup_avg = np.mean(pod_rbf_hprom_speedup)
    pod_rbf_hprom_speedup_max = np.max(pod_rbf_hprom_speedup)

    # Print the results
    print(f"HPROM average speedup: {hprom_speedup_avg:.2f}x, max speedup: {hprom_speedup_max:.2f}x")
    print(f"HRNM average speedup: {hrnm_speedup_avg:.2f}x, max speedup: {hrnm_speedup_max:.2f}x")
    print(f"POD-RBF HPROM average speedup: {pod_rbf_hprom_speedup_avg:.2f}x, max speedup: {pod_rbf_hprom_speedup_max:.2f}x")

if __name__ == "__main__":
    # Define file paths and grid data
    npz_file = 'rom_results_hprom.npz'
    output_folder = 'hrom_plots'

    num_cells_x, num_cells_y = 750, 750  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid

    # Define the mu1 and mu2 parameters
    mu1_values = [5.19, 4.56, 4.75]
    mu2_values = [0.026, 0.019, 0.02]

    # Dynamically generate snapshot file paths based on mu1 and mu2 values
    fom_snap_files = [f'hdm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    hprom_snap_files = [f'hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    hrnm_snap_files = [f'hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]
    pod_rbf_hprom_snap_files = [f'pod_rbf_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy' for mu1, mu2 in zip(mu1_values, mu2_values)]

    # Run the plotting and speedup calculation
    plot_npz_with_fom(npz_file, fom_snap_files, hprom_snap_files, hrnm_snap_files, pod_rbf_hprom_snap_files, grid_x, grid_y, output_folder)

