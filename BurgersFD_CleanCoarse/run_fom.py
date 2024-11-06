"""
Run the Full-Order Model (FOM) for the specified parameters and save the HDM snapshots.
"""

import time
import numpy as np
from hypernet2D import make_2D_grid, load_or_compute_snaps

def main(mu1=4.75, mu2=0.02, save_snaps=True):
    """
    Main function to run the Full-Order Model (FOM) and compute the HDM snapshots.
    
    Parameters:
    mu1 (float): First parameter for the FOM evaluation (e.g., inlet state).
    mu2 (float): Second parameter for the FOM evaluation (e.g., source term rate).
    save_snaps (bool): Whether to save the HDM snapshots or just return them.
    
    Returns:
    elapsed_time (float): Time taken to compute the HDM snapshots.
    hdm_snaps (np.ndarray): Array containing the HDM snapshots.
    """
    
    snap_folder = 'param_snaps'  # Folder where snapshots are stored

    # Time-stepping and grid setup for the 2D problem
    dt = 0.05  # Time step size
    num_steps = 500  # Number of time steps
    num_cells_x, num_cells_y = 250, 250  # Grid size in x and y directions
    xl, xu, yl, yu = 0, 100, 0, 100  # Domain boundaries
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)  # Create the 2D grid

    # Initial conditions for u and v (2D velocity components or state variables)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))  # Concatenate initial conditions

    # Parameter values for FOM evaluation
    mu_rom = [mu1, mu2]

    # Time-stepping to compute the HDM snapshots (FOM)
    t0 = time.time()
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
    elapsed_time = time.time() - t0
    print(f'Elapsed FOM time: {elapsed_time:.3e} seconds')

    # Save the HDM snapshots if needed
    if save_snaps:
        np.save(f'hdm_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', hdm_snaps)
        print(f'HDM snapshots saved as hdm_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy')

    # Return elapsed time and HDM snapshots
    return elapsed_time, hdm_snaps

# Run the FOM if the script is executed
if __name__ == "__main__":
    main()
