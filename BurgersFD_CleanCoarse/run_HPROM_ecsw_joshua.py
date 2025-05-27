"""
Build a parameterized ROM with a global ROB, and compare it to the HDM at an out-of-sample
point
"""

import glob
import pdb
import random
import time

import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from hypernet2D import compute_ECSW_training_matrix_2D, make_2D_grid, plot_snaps, \
    load_or_compute_snaps, inviscid_burgers_implicit2D_LSPG, POD, inviscid_burgers_res2D, \
    inviscid_burgers_exact_jac2D, inviscid_burgers_ecsw_fixed
from joblib import Parallel, delayed

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def main(mu1= 4.75, mu2=0.02, compute_ecsw = False):

    snap_folder = 'param_snaps'
    num_vecs = 95

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    # Query point
    mu_rom = [mu1, mu2]

    # ECSW sample point (we only choose one)
    mu_samples = [[4.25, 0.0225]]

    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    full_basis = np.load('basis.npy', allow_pickle=True)
    basis_trunc = full_basis[:, :num_vecs]

    # Perform ECSW hyper-reduction
    # Specify if you want to compute it or not (it's loaded below if you don't compute)
    if compute_ecsw:
        snap_sample_factor = 10
        Clist = []
        for imu, mu in enumerate(mu_samples):
            mu_snaps = all_snaps_list[imu]
            print('Generating training block for mu = {}'.format(mu))
            Ci = compute_ECSW_training_matrix_2D(mu_snaps[:, 3:num_steps:snap_sample_factor],
                                              mu_snaps[:, 0:num_steps - 3:snap_sample_factor],
                                              basis_trunc, inviscid_burgers_res2D,
                                              inviscid_burgers_exact_jac2D, grid_x, grid_y, dt, mu)
            Clist += [Ci]
        C = np.vstack(Clist)
        idxs = np.zeros((num_cells_y, num_cells_x))

        # Sampling all boundary points
        nn_xl = 1
        nn_xr = 1
        nn_y = 1
        idxs[nn_y:-nn_y, nn_xl:-nn_xr] = 1

        # Larger weighting for boundary terms due to the Dirichlet boundary condition
        bc_w = 50
        C_cor = bc_w * C[:, (idxs == 0).ravel()]
        C = C[:, (idxs == 1).ravel()]

        t1 = time.time()
        C = np.ascontiguousarray(C, dtype=np.float64)
        combined_weights = []
        res = Parallel(n_jobs=-1, verbose=10)(delayed(nnls)(c, c.sum(axis=1), maxiter=9999999999) for c in np.array_split(C,1,axis=1))
        for wi in res:
            combined_weights += [wi[0]]
        weights = np.hstack(combined_weights)

        #weights_boundary, res_boundary = nnls(C_cor, C_cor.sum(axis=1), maxiter=9999999999)
        print('nnls solve time: {}'.format(time.time() - t1))

        print('nnls solver residual: {}'.format(
            np.linalg.norm(C @ weights - C.sum(axis=1)) / np.linalg.norm(
                - C.sum(axis=1))))

        weights = weights.reshape((num_cells_y - 2 * nn_y, num_cells_x - (nn_xl + nn_xr)))
        full_weights = bc_w * np.ones((num_cells_y, num_cells_x))
        #full_weights = np.ones((num_cells_y, num_cells_x)) * weights.sum() / 100
        full_weights[idxs > 0] = weights.ravel()
        weights = full_weights.ravel()
        np.save('ecsw_weights_lspg', weights)
        plt.rcParams.update({
          "text.usetex": True,
          "mathtext.fontset": "stix",
          "font.family": ["STIXGeneral"]})
        plt.rc('font', size=16)
        plt.spy(weights.reshape((250, 250)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.title('PROM Reduced Mesh')
        plt.tight_layout()
        plt.savefig('prom-reduced-mesh.png', dpi=300)   
    else:
        weights = np.load('ecsw_weights_lspg.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))

    # Time-stepping to compute the HPROM at the out-of-sample parameter point
    t0 = time.time()
    rom_y, times = inviscid_burgers_ecsw_fixed(grid_x, grid_y, weights, w0, dt, num_steps, mu_rom, basis_trunc)
    jac_time, res_time, ls_time = times
    elapsed_time = time.time() - t0
    print(f'Elapsed HPROM time: {elapsed_time:.3e} seconds')

    # Load the corresponding HDM snapshots for comparison
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Compute the ROM snapshots
    rom_snaps = basis_trunc @ rom_y

    # Save the HPROM snapshots
    np.save(f'dd_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', rom_snaps)
    print(f'Snapshot saved as hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy')

    # Commented section for visualization (plotting)
    '''
    snaps_to_plot = range(0, 501, 100)  # Select snapshots at specific time intervals to plot
    fig, ax1, ax2 = plot_snaps(grid_x, grid_y, hdm_snaps, snaps_to_plot, label='HDM')
    plot_snaps(grid_x, grid_y, rom_snaps, snaps_to_plot, label='HPROM', fig_ax=(fig, ax1, ax2), color='blue', linewidth=1)

    # Add legends and save the plot
    ax1.legend(), ax2.legend()
    plt.tight_layout()
    plt.savefig('dd_hprom_mu_{:1.2e}_{:1.2e}.png'.format(mu_rom[0], mu_rom[1]), dpi=300)
    '''

    # Compute and print the relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Return elapsed time and relative error
    return elapsed_time, relative_error


if __name__ == "__main__":
    main(mu1 = 5.19, mu2= 0.026, compute_ecsw=False)
