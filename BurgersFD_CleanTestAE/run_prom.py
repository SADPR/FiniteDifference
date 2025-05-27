"""
Build a parameterized ROM with a global Reduced Order Basis (ROB) and compare it to the HDM at an out-of-sample point.
"""

import glob
import pdb
import time

import numpy as np
import matplotlib.pyplot as plt
from hypernet2D import make_2D_grid, plot_snaps, load_or_compute_snaps, inviscid_burgers_implicit2D_LSPG, POD
from config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

def get_snapshot_params():
  MU1_LOW, MU1_HIGH = MU1_RANGE
  MU2_LOW, MU2_HIGH = MU2_RANGE
  mu1_samples = np.linspace(MU1_LOW, MU1_HIGH, SAMPLES_PER_MU)
  mu2_samples = np.linspace(MU2_LOW, MU2_HIGH, SAMPLES_PER_MU)
  mu_samples = []
  for mu1 in mu1_samples:
    for mu2 in mu2_samples:
      mu_samples += [[mu1, mu2]]
  return mu_samples

def main(mu1=4.75, mu2=0.02, load_basis=True):
    """
    Main function to build and evaluate a ROM using a global ROB, and compare it to the HDM.
    
    Parameters:
    mu1 (float): First parameter for the ROM evaluation (e.g., inlet state).
    mu2 (float): Second parameter for the ROM evaluation (e.g., source term rate).
    load_basis (bool): Whether to load a precomputed basis or compute it from scratch.
    """
    
    snap_folder = 'param_snaps'  # Folder where snapshots are stored
    num_modes = 10  # Number of modes to keep after truncating the basis

    # Time-stepping and grid setup for the 2D problem
    dt = 0.05  # Time step size
    num_steps = 500  # Number of time steps
    num_cells_x, num_cells_y = 50, 50  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid

    # Initial conditions for u and v (2D velocity components or state variables)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))  # Concatenate initial conditions

    # Parameter values for ROM evaluation
    mu_rom = [mu1, mu2]

    # Load or compute the HDM snapshots and build the Reduced Order Basis (ROB)
    if load_basis:
        # Load a precomputed basis
        full_basis = np.load('basis.npy', allow_pickle=True)
        basis_trunc = full_basis[:, :num_modes]  # Truncate the basis to the desired number of modes
    else:
        '''
        # Compute the basis by collecting snapshots over a range of parameters
        all_snaps_list = []
        for mu in get_snapshot_params():
            # Generate or load HDM snapshots for each parameter combination
            snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
            all_snaps_list.append(snaps)

        snaps = np.hstack(all_snaps_list)  # Stack snapshots horizontally
        '''
        # Compute the basis by collecting snapshots over a range of parameters
        snap_count = len(get_snapshot_params())
        snapshot_shape = load_or_compute_snaps(get_snapshot_params()[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder).shape
        total_snaps = snapshot_shape[1] * snap_count  # Total number of columns (time steps * parameters)

        # Pre-allocate memory for all snapshots
        snaps = np.zeros((snapshot_shape[0], total_snaps))

        # Collect snapshots into the pre-allocated array
        col_offset = 0
        for mu in get_snapshot_params():
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
            snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu  # Insert directly
            col_offset += snap_mu.shape[1]  # Update column offset


        # Define the method for computing the POD
        pod_method = 'rsvd'  # Choose between 'svd' (standard SVD) or 'rsvd' (randomized SVD)

        # Perform POD to compute the ROB based on the selected method
        if pod_method == 'rsvd':
            # If using randomized SVD, specify the number of modes and optionally a random state
            t0 = time.time()
            basis, sigma = POD(snaps, num_modes=num_modes, method='rsvd')
            elapsed_time = time.time() - t0
            print(f'Elapsed SVD time: {elapsed_time:.3e} seconds')
        else:
            # Use standard SVD if no specific method is chosen or 'svd' is selected
            basis, sigma = POD(snaps, num_modes=num_modes, method='svd')

        # Save the singular values and the full basis
        np.save(f'sigma.npy', sigma[:num_modes])  # Save only the top num_modes singular values
        np.save(f'basis.npy', basis[:, :num_modes])  # Save the truncated basis (up to num_modes)

        # Truncate the basis for further use
        basis_trunc = basis[:, :num_modes]

    # Time-stepping to compute the Reduced-Order Model (ROM) at out-of-sample parameter point
    t0 = time.time()
    rom_snaps, times = inviscid_burgers_implicit2D_LSPG(grid_x, grid_y, w0, dt, num_steps, mu_rom, basis_trunc)
    elapsed_time = time.time() - t0
    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')

    # Load the corresponding HDM snapshots for comparison
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Save the ROM snapshots
    np.save(f'rom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', rom_snaps)
    print(f'Snapshot saved as rom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy')

    # Commented section for visualization (plotting)
    '''
    snaps_to_plot = range(0, 500, 100)  # Select snapshots at specific time intervals to plot
    fig, ax1, ax2 = plot_snaps(grid_x, grid_y, hdm_snaps, snaps_to_plot, label='HDM')
    plot_snaps(grid_x, grid_y, rom_snaps, snaps_to_plot, label='PROM', fig_ax=(fig, ax1, ax2), color='blue', linewidth=1)

    # Add legends and save the plot
    ax1.legend(), ax2.legend()
    plt.tight_layout()
    plt.savefig('prom_mu_{:1.2e}_{:1.2e}.png'.format(mu_rom[0], mu_rom[1]), dpi=300)
    '''

    # Compute and print the relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Return elapsed time and relative error
    return elapsed_time, relative_error

# Run the main function if the script is executed
if __name__ == "__main__":
    main(load_basis=True)

