"""
Run the autoencoder PROM, and compare it to the HDM at an out-of-sample
point
"""

import os
import time

from lsqnonneg import lsqnonneg
from scipy.optimize import nnls
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

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
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def load_autoencoder_monitor(model_path, device, plot=False):
  torch.set_default_dtype(torch.float32)
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

def main(mu1=4.56, mu2=0.019, compute_ecsw=False):

    model_path = 'autoenc.pt'
    snap_folder = 'param_snaps'

    # Query point of HPROM-ANN
    mu_rom = [mu1, mu2]

    # Sample point for ECSW
    mu_samples = [[4.25, 0.0225]]

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
              return basis @ x + basis2 @ rnm(x) #basis2 @ rnm(torch.cat((x, tmu)))
            else:
              with torch.no_grad():
                return basis @ x + basis2 @ rnm(x) #basis2 @ rnm(torch.cat((x, tmu)))

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

        # Select boundaries
        nn_x = 1
        nn_y = 1
        idxs[nn_y:-nn_y, nn_x:-nn_x] = 1
        C = C[:, (idxs == 1).ravel()]

        # Weighting for boundary
        bc_w = 10

        t1 = time.time()
        C = np.ascontiguousarray(C, dtype=np.float64)
        b = np.ascontiguousarray(C.sum(axis=1), dtype=np.float64)
        u,_,_,_= RandomizedSingularValueDecomposition().Calculate(C.T, 1e-8)
        hyper_reduction_element_selector = EmpiricalCubatureMethod()
        hyper_reduction_element_selector.SetUp(u, InitialCandidatesSet = None, constrain_sum_of_weights=True, constrain_conditions = False)
        hyper_reduction_element_selector.Run()
        num_elements = C.shape[1]
        weights = np.zeros(num_elements)
        # Assign weights at specific indices
        weights[hyper_reduction_element_selector.z] = hyper_reduction_element_selector.w
        print('ECM solve time: {}'.format(time.time() - t1))

        print('ECM solver residual: {}'.format(
          np.linalg.norm(C @ weights - b) / np.linalg.norm(b)))

        weights = weights.reshape((num_cells_y - 2*nn_y, num_cells_x - 2*nn_x))
        full_weights = bc_w*np.ones((num_cells_y, num_cells_x))
        full_weights[idxs > 0] = weights.ravel()
        weights = full_weights.ravel()
        np.save('ecsw_weights_rnm', weights)
        '''
        plt.clf()
        plt.rc('font', size=16)
        plt.spy(weights.reshape((250, 250)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.title('PROM-NN Reduced Mesh')
        plt.tight_layout()
        plt.savefig('prom-nn-reduced-mesh.png', dpi=300)
        plt.show()
        '''
    else:
        weights = np.load('ecsw_weights_rnm.npy')
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
            return basis @ x + basis2 @ rnm(x)
        else:
            with torch.no_grad():
                return basis @ x + basis2 @ rnm(x)

    # Compute the ROM snapshots using the decode function
    man_snaps = np.array([decode(torch.tensor(ys[:, i].copy(), dtype=torch.float), False).numpy() for i in range(ys.shape[1])]).T

    # Load the corresponding HDM snapshots for comparison
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Commented section for visualization (plotting)
    '''
    inds_to_plot = range(0, 501, 100)
    snaps_to_plot = [hdm_snaps, man_snaps]
    labels = ['HDM', 'HPROM-ANN']
    colors = ['black', 'green']
    linewidths = [2, 1]
    fig, ax1, ax2 = compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths)

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    save_path = f'hprom-ann_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}_n{sizes[0]}_nbar{sizes[1]-sizes[0]}.png'
    print(f'Saving as "{save_path}"')
    plt.savefig(save_path, dpi=300)
    plt.show()
    '''

    # Print timings for the steps
    print(f'rnm_its: {man_its:.2f}, rnm_jac: {man_jac:.2f}, rnm_res: {man_res:.2f}, rnm_ls: {man_ls:.2f}')

    # Save the HRNM snapshot to a file
    np.save(f'hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy', man_snaps)
    print(f'Snapshot saved as hrnm_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy')

    # Compute and print the relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - man_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Return elapsed time and relative error
    return elapsed_time, relative_error

if __name__ == "__main__":
    main(compute_ecsw=False)
