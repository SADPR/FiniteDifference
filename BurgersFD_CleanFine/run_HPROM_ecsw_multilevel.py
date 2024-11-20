"""
Build a parameterized ROM with a global ROB, and compare it to the HDM at an out-of-sample point
"""

import glob
import pdb
import random
import time

import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import sys
sys.path.append('../')

from hypernet2D import compute_ECSW_training_matrix_2D, make_2D_grid, plot_snaps, \
    load_or_compute_snaps, inviscid_burgers_implicit2D_LSPG, POD, inviscid_burgers_res2D, \
    inviscid_burgers_exact_jac2D, inviscid_burgers_ecsw_fixed

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def main(mu1=5.19, mu2=0.026, compute_ecsw=True, mu_samples=None, snap_sample_factor=10):

    snap_folder = 'param_snaps'
    num_vecs = 95

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 750, 750
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    mu_rom = [mu1, mu2]
    if mu_samples == None:
        mu_samples = [[4.25, 0.0225]]

    # Generate or retrieve HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    # Construct basis using mu_samples params
    full_basis = np.load('basis.npy', allow_pickle=True)
    basis_trunc = full_basis[:, :num_vecs]

    # Perform ECSW hyper-reduction
    if compute_ecsw:
        t0 = time.time()
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

        # Boundary condition handling
        idxs = np.zeros((num_cells_y, num_cells_x))
        nn_xl = 1
        nn_xr = 1
        nn_y = 1
        bc_w = 50
        idxs[nn_y:-nn_y, nn_xl:-nn_xr] = 1
        C_cor = bc_w * C[:, (idxs == 0).ravel()]
        C_interior = C[:, (idxs == 1).ravel()]

        # Adjust the right-hand side vector
        b_cor = C_cor.sum(axis=1)
        b_interior = C_interior.sum(axis=1)

        print('Time to construct training matrix: {}'.format(time.time() - t0))
        t1 = time.time()

        # Multilevel NNLS approach with 2 levels
        num_subdomains = 48  # Number of subdomains at level 1
        C_subdomains = np.array_split(C_interior, num_subdomains, axis=1)

        # Prepare start indices for subdomains
        start_indices = np.cumsum([0] + [C_j.shape[1] for C_j in C_subdomains[:-1]])

        # Level 1: Solve NNLS subproblems independently in parallel
        def solve_nnls_subproblem(C_i, start_idx):
            b_i = C_i.sum(axis=1)  # Right-hand side vector for subdomain
            w_i, res_i = nnls(C_i, b_i)
            # Store only the non-zero weights
            non_zero_indices = np.nonzero(w_i)[0]
            w_i_nz = w_i[non_zero_indices]
            # Map indices to global indices
            indices_i = np.arange(start_idx, start_idx + C_i.shape[1])
            indices_i_nz = indices_i[non_zero_indices]
            print(f'Subdomain starting at index {start_idx}, size of reduced mesh: {non_zero_indices.shape[0]}')
            return w_i_nz, indices_i_nz

        results = Parallel(n_jobs=-1, verbose=10)(delayed(solve_nnls_subproblem)(C_i, start_idx)
                                                  for C_i, start_idx in zip(C_subdomains, start_indices))

        subdomain_weights = [res[0] for res in results]
        subdomain_indices = [res[1] for res in results]

        # Level 2: Combine the non-zero columns from Level 1
        # Collect all non-zero weights and their indices
        all_non_zero_indices = np.concatenate(subdomain_indices)
        all_non_zero_weights = np.concatenate(subdomain_weights)
        # Extract the corresponding columns from the original C_interior matrix
        C_level2 = C_interior[:, all_non_zero_indices]
        # Build z_level2 (weights from Level 1)
        z_level2 = all_non_zero_weights
        # Compute b_level2 by multiplying C_level2 with z_level2
        b_level2 = C_level2 @ z_level2

        # Solve NNLS at Level 2
        print("Solving NNLS at Level 2")
        w_level2, res_level2 = nnls(C_level2, b_level2)
        # This will run the same function nnls(C_level2, b_level2) multiple times in parallel.

        # Map weights back to global indices (relative to C_interior)
        weights_interior = np.zeros(C_interior.shape[1])
        weights_interior[all_non_zero_indices] = w_level2

        # Handle boundary weights
        weights_boundary, res_boundary = nnls(C_cor, b_cor)
        print('nnls solve time: {}'.format(time.time() - t1))
        print('Level 2 nnls residual: {}'.format(res_level2))
        print('nnz(weights_interior): {}'.format(np.sum(weights_interior > 0)))
        print('nnz(weights_boundary): {}'.format(np.sum(weights_boundary > 0)))

        # Assemble full weights vector
        weights_full = np.zeros(C.shape[1])
        # Map weights_interior back to full domain indices
        interior_indices = np.where(idxs.ravel() == 1)[0]
        weights_full[interior_indices] = weights_interior
        # Assign boundary weights
        boundary_indices = np.where(idxs.ravel() == 0)[0]
        weights_full[boundary_indices] = weights_boundary

        # Reshape weights
        weights = weights_full.reshape((num_cells_y, num_cells_x))
        np.save('ecsw_weights_multilevel', weights)
        plt.clf()
        plt.rcParams.update({
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": ["STIXGeneral"]})
        plt.rc('font', size=16)
        plt.spy(weights)
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.tight_layout()
        plt.savefig('ecsw_weights_multilevel.png', dpi=300)
    else:
        weights = np.load('ecsw_weights_multilevel.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))

    # Evaluate ROM at mu_rom
    t0 = time.time()
    weights_vector = weights.ravel()
    rom_y, times = inviscid_burgers_ecsw_fixed(grid_x, grid_y, weights_vector, w0, dt, num_steps, mu_rom, basis_trunc)
    jac_time, res_time, ls_time = times
    print('Elapsed time: {:3.3e}'.format(time.time() - t0))
    print('lspg_jac: {:3.2f}, lspg_res: {:3.2f}, lspg_ls: {:3.2f}'.format(jac_time, res_time, ls_time))
    rom_snaps = basis_trunc @ rom_y
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Plotting results
    snaps_to_plot = range(0, 501, 100)
    fig, ax1, ax2 = plot_snaps(grid_x, grid_y, hdm_snaps, snaps_to_plot, label='HDM')
    plot_snaps(grid_x, grid_y, rom_snaps, snaps_to_plot,
               label='HPROM', fig_ax=(fig, ax1, ax2), color='blue', linewidth=1)
    ax1.legend(), ax2.legend()
    ax1.grid(which='both')
    ax1.minorticks_on()
    ax2.grid(which='both')
    ax2.minorticks_on()
    plt.tight_layout()
    plt.savefig('predict_mu_{:1.2e}_{:1.2e}_hprom.png'.format(mu_rom[0], mu_rom[1]), dpi=300)
    plt.show()

    # Compute error
    print('Relative error: {:3.2f}%'.format(100 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)))
    mse = sum([np.linalg.norm(hdm_snaps[:, c] - rom_snaps[:, c])
               for c in range(hdm_snaps.shape[1])]) / sum(np.linalg.norm(hdm_snaps[:, c])
                                                          for c in range(hdm_snaps.shape[1]))
    print('MSE: {:3.2f}%'.format(100 * mse))
    return mse

if __name__ == "__main__":
    main()
