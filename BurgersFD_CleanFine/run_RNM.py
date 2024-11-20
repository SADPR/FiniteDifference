"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import glob
import pdb

import numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from hypernet2D import (load_or_compute_snaps, make_2D_grid, inviscid_burgers_implicit2D_LSPG,
                      inviscid_burgers_rnm2D, plot_snaps, POD)
from models import RNM_NN
from train_utils import TrainingMonitor
from config import DT, NUM_STEPS, NUM_CELLS, GRID_X, GRID_Y, W0

CARLBERG = False  # whether or not to use architecture from paper

from train_utils import get_data, show_model, TrainingMonitor
from train_autoencoder import (
                get_snapshot_params,
                ROM_SIZE,
                LR_INIT, LR_PATIENCE, COMPLETION_PATIENCE,
                MODEL_PATH
                )
from config import SEED, NUM_CELLS

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=16)

def load_autoencoder_monitor(model_path, device, plot=False):
  torch.set_default_dtype(torch.float32)
  rng = torch.Generator()
  rng = rng.manual_seed(SEED)
  np_rng = np.random.default_rng(SEED)

  sizes = np.load('sizes.npy', allow_pickle=True)

  print(sizes)
  rnm = RNM_NN(sizes[0], sizes[1]-sizes[0]).to(device)
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

def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(GRID_X, GRID_Y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

def main(mu1=4.75, mu2=0.02):

    model_path = 'autoenc.pt'
    snap_folder = 'param_snaps'

    mu_rom = [mu1, mu2]


    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    # load autoencoder
    rnm = load_autoencoder(model_path, device='cpu')

    # evaluate ROM at mu_rom
    # rom_snaps, times = inviscid_burgers_LSPG(GRID, W0, DT, NUM_STEPS, mu_rom, basis_trunc)
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
    # hdm_snaps = load_or_compute_snaps(mu_rom, GRID, W0, DT, NUM_STEPS, snap_folder=snap_folder)
    # proj_snaps = project_onto_autoencoder(hdm_snaps, auto, ref)
    sizes = np.load('sizes.npy', allow_pickle=True)
    full_basis = np.load('basis.npy', allow_pickle=True)
    ref = numpy.zeros_like(full_basis[:, 0]).squeeze()
    basis = torch.tensor(full_basis[:, :sizes[0]], dtype=torch.float)
    basis2 = torch.tensor(full_basis[:, sizes[0]:sizes[1]], dtype=torch.float)
    # basis2[:, 130:-1] = 0

    # Time-stepping to compute the Reduced-Order Model (ROM) using RNM
    t0 = time.time()
    rnm_snaps, rnm_times = inviscid_burgers_rnm2D(grid_x, grid_y, w0, dt, num_steps, mu_rom, rnm, ref, basis, basis2)
    elapsed_time = time.time() - t0
    rnm_its, rnm_jac, rnm_res, rnm_ls = rnm_times

    print(f'Elapsed time: {elapsed_time:.3e} seconds')

    # Save the snapshot to a file
    np.save(f'rnm_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', rnm_snaps)
    print(f'Snapshot saved as rnm_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy')

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - rnm_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Print RNM times
    print(f'rnm_its: {rnm_its:.2f}, rnm_jac: {rnm_jac:.2f}, rnm_res: {rnm_res:.2f}, rnm_ls: {rnm_ls:.2f}')

    # Plot and compare snapshots (currently commented out)

    inds_to_plot = range(0, num_steps + 1, 100)
    snaps_to_plot = [hdm_snaps, rnm_snaps]
    labels = ['HDM', 'PROM-NN']
    colors = ['black', 'green']
    linewidths = [2, 1]
    compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)

    plt.tight_layout()
    plt.grid()
    plt.legend(loc=2)
    plt.savefig(f'plot_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}_n{sizes[0]}_nbar{sizes[1]-sizes[0]}.png', dpi=300)
    

    # Return elapsed time and relative error
    return elapsed_time, relative_error


if __name__ == "__main__":
    main()