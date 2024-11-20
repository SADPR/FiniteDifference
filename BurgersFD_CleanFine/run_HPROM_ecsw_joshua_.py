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
from joblib import Parallel, delayed

import sys
sys.path.append('../')

from hypernet2D import compute_ECSW_training_matrix_2D, make_2D_grid, plot_snaps, \
    load_or_compute_snaps, inviscid_burgers_implicit2D_LSPG, POD, inviscid_burgers_res2D, \
    inviscid_burgers_exact_jac2D, inviscid_burgers_ecsw_fixed
#from lsqnonneg import lsqnonneg

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def main(mu1=5.19, mu2=0.026, compute_ecsw=True, mu_samples=None, snap_sample_factor=10):

    snap_folder = 'param_snaps'
    num_vecs = 95

    # dt = 0.07
    # num_steps = 500
    # num_cells = 500
    # xl, xu = 0, 100
    # w0 = np.ones(num_cells)
    # grid = make_1D_grid(xl, xu, num_cells)
    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
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
    # ###################################
    # # mu_samples = [mu_rom]
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]
    #
    # snaps = np.hstack(all_snaps_list)

    # construct basis using mu_samples params
    # basis, sigma = POD(snaps)
    # basis_trunc = basis[:, :num_vecs]
    full_basis = np.load('basis.npy', allow_pickle=True)
    basis_trunc = full_basis[:, :num_vecs]

    # Perform ECSW hyper-reduction
#    compute_ecsw = False
    if compute_ecsw:
        ecsw_max_support = 5000
        ecsw_err_thresh = 0.00594
#        snap_sample_factor = 10
        t0 = time.time()
        Clist = []
        for imu, mu in enumerate(mu_samples):
            mu_snaps = all_snaps_list[imu]
            print('Generating training block for mu = {}'.format(mu))
            Ci = compute_ECSW_training_matrix_2D(mu_snaps[:, 3:num_steps:snap_sample_factor],
                                              mu_snaps[:, 0:num_steps - 3:snap_sample_factor],
                                              basis_trunc, inviscid_burgers_res2D,
                                              inviscid_burgers_exact_jac2D, grid_x, grid_y, dt, mu)
            plt.imshow(Ci)
            plt.show()
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

        # t1 = time.time()
        # C_cor[:, :] = 0
        # weights, rnormsq, res = lsqnonneg(C, C.sum(axis=1) - C_cor.sum(axis=1), max_support=ecsw_max_support,
        #                                   rel_err_thresh=ecsw_err_thresh)
        # print('Time to solve lsqnonneg: {}'.format(time.time() - t1))

        print('Time to construct training matrix: {}'.format(time.time() - t0))

        t1 = time.time()

        #Splitting up C
        combined_weights = []
        res = Parallel(n_jobs=-1, verbose=10)(delayed(nnls)(c, c.sum(axis=1), maxiter=9999999999) for c in np.array_split(C,20,axis=1))
        for wi in res:
            combined_weights += [wi[0]]
        weights = np.hstack(combined_weights)

#        weights, res = nnls(C, C.sum(axis=1))
        weights_boundary, res_boundary = nnls(C_cor, C_cor.sum(axis=1), maxiter=9999999999)
        print('nnls solve time: {}'.format(time.time() - t1))
        print('nnls res: {}\nnnls boundary res: {}'.format(res, res_boundary))
        print('nnz(weights): {}'.format(np.sum(weights > 0)))
        print('nnz(boundary weights): {}'.format(np.sum(weights_boundary > 0)))


#        C_cor[:, :] = 0
#        weights, _ = nnls(C, C.sum(axis=1) - C_cor.sum(axis=1), maxiter=99999999)
#        print('nnls solve time: {}'.format(time.time() - t1))

        print('nnls solver residual: {}'.format(
            np.linalg.norm(C @ weights - C.sum(axis=1)) / np.linalg.norm(
                - C.sum(axis=1))))

        # weights = weights.reshape((num_cells_y, num_cells_x - nn_x))
        # weights = np.concatenate((bc_w*np.ones((num_cells_y, nn_x)), weights), axis=1)
        # weights = weights.ravel()
        weights = weights.reshape((num_cells_y - 2 * nn_y, num_cells_x - (nn_xl + nn_xr)))
        #full_weights = np.zeros((num_cells_y, num_cells_x))
        full_weights = np.ones((num_cells_y, num_cells_x)) * weights.sum() / 100
        full_weights[idxs > 0] = weights.ravel()
        #full_weights[~(idxs > 0)] = weights_boundary.ravel()
        # weights = np.concatenate((np.ones((num_cells_y, nn)), weights), axis=1)
        weights = full_weights.ravel()
        np.save('ecsw_weights_lspg', weights)
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
        weights = np.load('ecsw_weights_lspg.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))
    # weights = np.append(np.ones((nn,)), weights1)
    # print(weights1)
    # plt.plot(weights)
    # plt.show()

    # evaluate ROM at mu_rom
    t0 = time.time()
    # rom_y = inviscid_burgers_ecsw(grid_x, grid_y, weights, w0, dt, num_steps, mu_rom, basis_trunc)
    rom_y, times = inviscid_burgers_ecsw_fixed(grid_x, grid_y, weights, w0, dt, num_steps, mu_rom, basis_trunc)
    # rom_snaps, times = inviscid_burgers_implicit2D_LSPG(grid_x, grid_y, w0, dt, num_steps, mu_rom, basis_trunc)
    jac_time, res_time, ls_time = times
    print('Elapsed time: {:3.3e}'.format(time.time() - t0))
    print('lspg_jac: {:3.2f}, lspg_res: {:3.2f}, lspg_ls: {:3.2f}'.format(jac_time, res_time, ls_time))
    rom_snaps = basis_trunc @ rom_y
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # fig, ax = plt.subplots()
    # snaps_to_plot = range(int(num_steps / 10), num_steps + 1, int(np.ceil(2 * num_steps / 10)))
    snaps_to_plot = range(0, 501, 100)
    fig, ax1, ax2 = plot_snaps(grid_x, grid_y, hdm_snaps, snaps_to_plot,
               label='HDM')
    plot_snaps(grid_x, grid_y, rom_snaps, snaps_to_plot,
               label='HPROM', fig_ax=(fig, ax1, ax2), color='blue', linewidth=1)

    # ax.set_xlim([grid.min(), grid.max()])
    # ax.set_xlabel('x')
    # ax.set_ylabel('w')
    # plt.title('Comparing HDM and ROM')
    ax1.legend(), ax2.legend()
    ax1.grid(which='both')
    #ax1.tick_params(axis='y', which='both')
    ax1.minorticks_on()  
    ax2.grid(which='both')
    #ax2.tick_params(axis='y', which='both')
    ax2.minorticks_on()
    plt.tight_layout()
    import pickle
    with open('predict_mu_{:1.2e}_{:1.2e}_hprom.pickle'.format(mu_rom[0], mu_rom[1]), 'wb') as fid:
        pickle.dump(fig, fid)
    plt.tight_layout()
    plt.savefig('predict_mu_{:1.2e}_{:1.2e}_hprom.png'.format(mu_rom[0], mu_rom[1]), dpi=300)
    plt.show()
    print('Relative error: {:3.2f}%'.format(100 * np.linalg.norm(hdm_snaps - rom_snaps) / np.linalg.norm(hdm_snaps)))
    mse = sum([np.linalg.norm(hdm_snaps[:, c] - rom_snaps[:, c])
               for c in range(hdm_snaps.shape[1])]) / sum(np.linalg.norm(hdm_snaps[:, c])
                                                          for c in range(hdm_snaps.shape[1]))
    print('MSE: {:3.2f}%'.format(100*mse))
#    err = [np.linalg.norm(hdm_snaps[:, c] - rom_snaps[:, c]) / np.linalg.norm(hdm_snaps[:, c])
#           for c in range(hdm_snaps.shape[1])]
#    plt.plot(dt * np.arange(hdm_snaps.shape[1]), err)
#    plt.xlabel('$t$')
#    plt.ylabel('$\\left| \\tilde{\\mathbf{u}} - \\mathbf{u} \\right| / \\left| \\mathbf{u} \\right|$')
#    plt.title('HPROM: relative error as a function of time')
#    plt.grid()
#    plt.tight_layout()
#    plt.savefig('re_lspg_mu_{:1.2e}_{:1.2e}.png'.format(*mu_rom), dpi=300)
#    np.save('err_hprom', np.array(err))
#    plt.sho1w()
    return mse

if __name__ == "__main__":
    main()
