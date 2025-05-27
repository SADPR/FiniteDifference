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

def main(mu1=4.875, mu2=0.0225, compute_ecsw=False):

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

        nn_xl = 1
        nn_xr = 1
        nn_y = 1
        bc_w = 50
        idxs[nn_y:-nn_y, nn_xl:-nn_xr] = 1
        # idxs[:, nn_xl:] = 1
        C_cor = bc_w * C[:, (idxs == 0).ravel()]
        C = C[:, (idxs == 1).ravel()]

        t1 = time.time()

        #Splitting up C
        combined_weights = []
        res = Parallel(n_jobs=4, verbose=10)(delayed(nnls)(c, c.sum(axis=1), maxiter=9999999999) for c in np.array_split(C,24,axis=1))
        for wi in res:
            combined_weights += [wi[0]]
        weights = np.hstack(combined_weights)

        print('nnls solver residual: {}'.format(
            np.linalg.norm(C @ weights - C.sum(axis=1)) / np.linalg.norm(
                - C.sum(axis=1))))
        
        print('nnls solve time: {}'.format(time.time() - t1))

        print(np.nonzero(weights)[0].shape)        
        weights = weights.reshape((num_cells_y - 2 * nn_y, num_cells_x - (nn_xl + nn_xr)))
        full_weights = bc_w * np.ones((num_cells_y, num_cells_x))
        #full_weights = np.ones((num_cells_y, num_cells_x)) * weights.sum() / 100
        full_weights[idxs > 0] = weights.ravel()
        weights = full_weights.ravel()
        np.save('ecsw_weights_hrnm_domain_decomposition', weights)
        plt.clf()
        plt.rcParams.update({
          "text.usetex": True,
          "mathtext.fontset": "stix",
          "font.family": ["STIXGeneral"]})
        plt.rc('font', size=16)
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
#        plt.title('PROM Reduced Mesh')
        plt.tight_layout()
        plt.savefig('prom-reduced-mesh.png', dpi=300)
    else:
        #weights = np.load('pod_rbf_global_model/ecm_weights_rbf_global.npy')
        weights = np.load('ecsw_weights_rnm_working.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))
    #END ECSW

    # Time-stepping to compute the HPROM-ANN at the out-of-sample parameter point
    t0 = time.time()
    q_snaps = basis.T@hdm_snaps
    ys, man_times = inviscid_burgers_rnm2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu_rom_backup, rnm, ref, basis, basis2, weights)
    man_its, man_jac, man_res, man_ls = man_times
    elapsed_time = time.time() - t0
    print(f'Elapsed HPROM-ANN time: {elapsed_time:.3e} seconds')

    # Define function for decoding
    tmu = torch.tensor(mu_rom.copy(), dtype=torch.float)

    def decode(x, with_grad=True):
        if with_grad:
            return basis @ x + basis2 @ rnm(torch.cat((x, tmu)))#rnm(x)
        else:
            with torch.no_grad():
                return basis @ x + basis2 @ rnm(torch.cat((x, tmu)))#rnm(x)

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
    save_path = f'pod_ann_hprom_snaps_mu1_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}_n{sizes[0]}_nbar{sizes[1]-sizes[0]}.png'
    print(f'Saving as "{save_path}"')
    plt.savefig(save_path, dpi=300)
    plt.show()
    

    # Print timings for the steps
    print(f'rnm_its: {man_its:.2f}, rnm_jac: {man_jac:.2f}, rnm_res: {man_res:.2f}, rnm_ls: {man_ls:.2f}')

    # Save the HRNM snapshot to a file
    np.save(f'pod_ann_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy', man_snaps)
    print(f'Snapshot saved as pod_ann_hprom_snaps_mu1_{mu1:.2f}_mu2_{mu2:.3f}.npy')

    # Compute and print the relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - man_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    ############################################################################
    # ADDED CODE: Create animation overlaying HDM and POD-RBF using FuncAnimation
    ############################################################################
    import matplotlib.animation as animation

    # We'll create a figure and reuse it for each frame
    fig_anim, (ax1_anim, ax2_anim) = plt.subplots(2, 1, figsize=(10, 8))

    # We define the data sets and labeling for overlay
    snaps_to_plot_anim = [hdm_snaps, man_snaps]
    labels_anim = ['HDM', 'POD-ANN HPROM']
    colors_anim = ['black', 'green']
    linewidths_anim = [2, 2]

    def animate_func(frame_idx):
        print(f"Processing frame {frame_idx + 1}...")
        ax1_anim.clear()
        ax2_anim.clear()

        # Fix the y-limits for both subplots
        ax1_anim.set_ylim(0, 6.5)
        ax2_anim.set_ylim(0, 6.5)

        # Overlay both HDM & POD-RBF for this single time index 'frame_idx'
        for i, each_snaps in enumerate(snaps_to_plot_anim):
            plot_snaps(
                grid_x, grid_y,
                each_snaps, [frame_idx],  # single time index
                label=labels_anim[i],
                fig_ax=(fig_anim, ax1_anim, ax2_anim),
                color=colors_anim[i],
                linewidth=linewidths_anim[i]
            )
        ax1_anim.legend()
        ax2_anim.legend()
        ax1_anim.set_title(f'Timestep = {frame_idx}')

    save_gif = False
    if save_gif:
        anim = animation.FuncAnimation(
            fig_anim, animate_func,
            frames=range(num_steps),
            interval=300,  # ms between frames
            blit=False,
            repeat=False
        )

        # Save the animation as a GIF with 30 FPS
        anim_filename = f'pod_ann_hprom_global_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.gif'
        anim.save(anim_filename, writer='imagemagick', fps=30)
        print(f"Saved animation '{anim_filename}' with overlay of HDM & POD-RBF at each timestep.")

    # Return elapsed time and relative error
    return elapsed_time, relative_error

if __name__ == "__main__":
    main(compute_ecsw=False)
