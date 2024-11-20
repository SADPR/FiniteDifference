"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import os
import time

from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import glob
import pdb

import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from hypernet2D import (load_or_compute_snaps, make_2D_grid,
                        plot_snaps, POD, inviscid_burgers_rnm2D_ecsw,
                        inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
                        compute_ECSW_training_matrix_2D_rnm)
from models import RNM_NN
from config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

from train_utils import TrainingMonitor
from train_autoencoder import (
                get_snapshot_params,
                LR_INIT, LR_PATIENCE, COMPLETION_PATIENCE,
                MODEL_PATH
                )
from config import SEED

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def load_autoencoder_monitor(model_path, device, plot=False):
  torch.set_default_dtype(torch.float32)
  sizes = np.load('sizes.npy', allow_pickle=True)

  print(sizes)
  rnm = RNM_NN(sizes[0]+2, sizes[1]-sizes[0]).to(device)
  opt = optim.Adam(rnm.parameters(), lr=LR_INIT)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1,
                                                   patience=LR_PATIENCE, verbose=True)

  monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE, rnm, opt, scheduler, train=False)
  monitor.load_from_path(model_path)
  return monitor

def load_autoencoder(model_path, device='cpu'):
  monitor = load_autoencoder_monitor(model_path, device, plot=False)
  return monitor.model

def project_onto_autoencoder(snaps, auto, ref):
  snaps_deref = snaps - np.expand_dims(ref, 1)
  snaps_t = torch.Tensor(snaps_deref.T)
  with torch.no_grad():
    out = auto(snaps_t.unsqueeze(1))
  out_np = out.squeeze().numpy().T
  out_ref = out_np + np.expand_dims(ref, 1)
  return out_ref


def compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(grid_x, grid_y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

  return fig, ax1, ax2

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

def main(mu1=4.75, mu2=0.02, compute_ecsw=False):

    model_path = 'autoenc.pt'
    snap_folder = 'param_snaps'

    # Query point of HPROM-ANN
    mu_rom = [mu1, mu2]

    # Sample point for ECSW
    mu_samples = [[4.25, 0.0225]]

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 750, 750
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    # load autoencoder
    rnm = load_autoencoder(model_path, device='cpu')

    # evaluate ROM at mu_rom
    mu_rom_backup = [v for v in mu_rom]
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
    sizes = np.load('sizes.npy', allow_pickle=True)
    full_basis = np.load('basis.npy', allow_pickle=True)
    ref = numpy.zeros_like(full_basis[:, 0]).squeeze()
    basis = torch.tensor(full_basis[:, :sizes[0]], dtype=torch.float)
    basis2 = torch.tensor(full_basis[:, sizes[0]:sizes[1]], dtype=torch.float)
    tmu = torch.tensor(mu_rom.copy(), dtype=torch.float)

    #ECSW
    if compute_ecsw:
        snap_sample_factor = 10

        Clist = []
        for imu, mu in enumerate(mu_samples):
          tmu = torch.tensor(mu.copy(), dtype=torch.float)
          mu_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
          
          def decode(x, with_grad=True):
            if with_grad:
              return  basis @ x + basis2 @ rnm(torch.cat((x, tmu))) #basis @ x + basis2 @ rnm(x)
            else:
              with torch.no_grad():
                return basis @ x + basis2 @ rnm(torch.cat((x, tmu))) #basis @ x + basis2 @ rnm(x)

          import functorch
          jacfwdfunc = functorch.jacfwd(decode)
          print('Generating training block for mu = {}'.format(mu))
          Ci = compute_ECSW_training_matrix_2D_rnm(mu_snaps[:, 2*imu+3:num_steps:snap_sample_factor],
                                                   mu_snaps[:, 2*imu+0:num_steps - 3 - 2*imu:snap_sample_factor],
                                                   basis, decode, jacfwdfunc, inviscid_burgers_res2D,
                                                   inviscid_burgers_exact_jac2D, grid_x, grid_y, dt, mu)
          Clist += [Ci]

        C = np.vstack(Clist)
        idxs = np.zeros((num_cells_y, num_cells_x))

        # Select interior nodes (excluding boundaries)
        nn_x = 1
        nn_y = 1
        idxs[nn_y:-nn_y, nn_x:-nn_x] = 1

        # Weighting for boundary
        bc_w = 10
        
        C_cor = bc_w * C[:, (idxs == 0).ravel()]
        C_interior = C[:, (idxs == 1).ravel()]

        # Adjust the right-hand side vector
        b_cor = C_cor.sum(axis=1)

        t1 = time.time()

        # Multilevel NNLS approach with 2 levels
        num_subdomains = 24  # Number of subdomains at level 1
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
        print(C_level2.shape)
        w_level2, res_level2 = nnls(C_level2, b_level2, atol=1e-8)
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
        print('nnls solver residual: {}'.format(
            np.linalg.norm(C @ weights_full - C.sum(axis=1)) / np.linalg.norm(
                - C.sum(axis=1))))

        # Reshape weights
        weights = weights_full
        np.save('ecsw_weights_rnm_multilevel', weights)
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
        plt.savefig('ecsw_weights_rnm_multilevel.png', dpi=300)
    else:
        weights = np.load('ecsw_weights_rnm_multilevel.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))
    #END ECSW

    # Time-stepping to compute the HPROM-ANN at the out-of-sample parameter point
    t0 = time.time()
    ys, man_times = inviscid_burgers_rnm2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu_rom_backup, rnm, ref, basis, basis2, weights)
    man_its, man_jac, man_res, man_ls = man_times
    elapsed_time = time.time() - t0
    print(f'Elapsed HPROM-ANN time: {elapsed_time:.3e} seconds')

    # Define function for decoding
    tmu = torch.tensor(mu_rom.copy(), dtype=torch.float)

    def decode(x, with_grad=True):
        if with_grad:
            return basis @ x + basis2 @ rnm(torch.cat((x, tmu)))
        else:
            with torch.no_grad():
                return basis @ x + basis2 @ rnm(torch.cat((x, tmu)))

    # Compute the ROM snapshots using the decode function
    man_snaps = np.array([decode(torch.tensor(ys[:, i].copy(), dtype=torch.float), False).numpy() for i in range(ys.shape[1])]).T

    # Load the corresponding HDM snapshots for comparison
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Commented section for visualization (plotting)
    
    inds_to_plot = range(0, 501, 100)
    snaps_to_plot = [hdm_snaps, man_snaps]
    labels = ['HDM', 'HPROM-ANN']
    colors = ['black', 'green']
    linewidths = [2, 1]
    fig, ax1, ax2 = compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths)

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    save_path = f'dd_multilevel_hprom-ann_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}_n{sizes[0]}_nbar{sizes[1]-sizes[0]}.png'
    print(f'Saving as "{save_path}"')
    plt.savefig(save_path, dpi=300)
    plt.show()
    

    # Print timings for the steps
    print(f'rnm_its: {man_its:.2f}, rnm_jac: {man_jac:.2f}, rnm_res: {man_res:.2f}, rnm_ls: {man_ls:.2f}')

    # Save the HRNM snapshot to a file
    np.save(f'dd_multilevel_hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy', man_snaps)
    print(f'Snapshot saved as hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy')

    # Compute and print the relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - man_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Return elapsed time and relative error
    return elapsed_time, relative_error

if __name__ == "__main__":
    main(compute_ecsw=True)
