import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import nnls
from joblib import Parallel, delayed

from hypernet2D import (
    load_or_compute_snaps, plot_snaps,
    inviscid_burgers_pod_rbf_2D_global_ecsw,
    inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
    compute_ECSW_training_matrix_2D_rbf_global,
    decode_rbf_global
)
from config import DT, NUM_STEPS, NUM_CELLS, GRID_X, GRID_Y, W0

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i, snaps in enumerate(snaps_to_plot):
        plot_snaps(GRID_X, GRID_Y, snaps, inds_to_plot,
                   label=labels[i],
                   fig_ax=(fig, ax1, ax2),
                   color=colors[i],
                   linewidth=linewidths[i])
    return fig, ax1, ax2

def main(mu1=4.75, mu2=0.02, compute_ecsw=False, save_npy=False, save_plot=False):
    # Paths and parameters
    snap_folder = 'param_snaps'

    # Query point for the POD-RBF PROM
    mu_rom = [mu1, mu2]

    # Load the HDM snapshots for the query parameter combination
    hdm_snaps = load_or_compute_snaps(mu_rom, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)

    # Load global RBF weights and training data
    model_dir = "pod_rbf_global_model"
    try:
        with open(os.path.join(model_dir, 'global_weights.pkl'), 'rb') as f:
            data = pickle.load(f)
            W_global = data['W']  # Global weight matrix
            q_p_train = data['q_p_train']
            q_s_train = data['q_s_train']
            epsilon = data['epsilon']
            kernel_name = data['kernel_name']
        print("Loaded global training data successfully.")
    except FileNotFoundError:
        print(f"Training data file '{model_dir}/global_weights.pkl' not found.")
        exit(1)

    # Load the scaler
    try:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded Min-Max scaler successfully.")
    except FileNotFoundError:
        print(f"Scaler file '{model_dir}/scaler.pkl' not found.")
        exit(1)

    # Load the POD basis matrices
    try:
        U_p = np.load(os.path.join(model_dir, 'U_p.npy'))
        U_s = np.load(os.path.join(model_dir, 'U_s.npy'))
        print("Loaded POD basis matrices (U_p and U_s) successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print(f"mu1: {mu1}, epsilon: {epsilon}, kernel: {kernel_name}")

    # ECSW computation
    if compute_ecsw:
        snap_sample_factor = 8

        Clist = []
        for imu, mu in enumerate([[4.25, 0.0225]]):  # ECSW sample points
            mu_snaps = load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)
            prev_snaps = mu_snaps[:, :-1:snap_sample_factor]
            snaps = mu_snaps[:, 1::snap_sample_factor]

            print(f'Generating training block for mu = {mu}')
            Ci = compute_ECSW_training_matrix_2D_rbf_global(
                snaps, prev_snaps, U_p, U_s, W_global, q_p_train, q_s_train, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D, 
                GRID_X, GRID_Y, DT, mu, scaler, epsilon, kernel_type=kernel_name
            )
            Clist.append(Ci)

        C = np.vstack(Clist)
        print("Full C shape:", C.shape)

        # Create mask for interior cells (boundaries not treated specially)
        idxs = np.zeros((NUM_CELLS, NUM_CELLS))
        nn_xl = 1
        nn_xr = 1
        nn_y = 1
        bc_w = 50  # This constant will be used for the boundary
        idxs[nn_y:-nn_y, nn_xl:-nn_xr] = 1

        # Split C into interior only (discarding boundary columns for NNLS)
        C_interior = C[:, (idxs == 1).ravel()]
        print("Interior part shape:", C_interior.shape)

        t1 = time.time()

        # Partition interior into subdomains for multilevel NNLS
        num_subdomains = 20
        C_subdomains = np.array_split(C_interior, num_subdomains, axis=1)

        subdomain_weights = []
        subdomain_indices = []
        for i, C_i in enumerate(C_subdomains):
            print(f"Solving NNLS for subdomain {i+1}/{num_subdomains}")
            b_i = C_i.sum(axis=1)
            w_i, _ = nnls(C_i, b_i, maxiter=9999999999)
            nz_idx = np.nonzero(w_i)[0]
            print(f"Subdomain {i+1} selected {nz_idx.shape[0]} elements")
            subdomain_weights.append(w_i[nz_idx])
            start_idx = sum(C_j.shape[1] for C_j in C_subdomains[:i])
            indices_i = np.arange(start_idx, start_idx + C_i.shape[1])
            subdomain_indices.append(indices_i[nz_idx])

        # Level 2: Merge nonzero selections from all subdomains
        all_non_zero_indices = np.concatenate(subdomain_indices)
        all_non_zero_weights = np.concatenate(subdomain_weights)
        print("Total nonzero indices from Level 1:", all_non_zero_indices.shape)
        print("Total nonzero weights from Level 1:", all_non_zero_weights.shape)

        C_level2 = C_interior[:, all_non_zero_indices]
        b_level2 = C_level2 @ all_non_zero_weights
        print("Level 2: C_level2 shape:", C_level2.shape, "b_level2 shape:", b_level2.shape)
        w_level2, res_level2 = nnls(C_level2, b_level2, maxiter=9999999999, atol = 1e-10)
        print("Level 2 NNLS nonzero count:", np.sum(w_level2 > 0))
        print("Level 2 NNLS residual:", res_level2)
        
        print('nnls solve time: {}'.format(time.time() - t1))

        weights_interior = np.zeros(C_interior.shape[1])
        weights_interior[all_non_zero_indices] = w_level2

        # Assemble full weight vector:
        # For interior positions, use the computed interior weights;
        # for boundary positions, fill with the constant bc_w.
        full_weights = bc_w * np.ones((NUM_CELLS, NUM_CELLS))
        interior_indices = np.where(idxs.ravel() == 1)[0]
        full_weights.ravel()[interior_indices] = weights_interior

        weights = full_weights.ravel()
        np.save(os.path.join(model_dir, 'ecsw_weights_rbf_global.npy'), weights)
        plt.clf()
        plt.rcParams.update({
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": ["STIXGeneral"]})
        plt.rc('font', size=16)
        plt.spy(weights.reshape((NUM_CELLS, NUM_CELLS)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.tight_layout()
        plt.savefig('ecsw_rbf_global_reduced_mesh.png', dpi=300)

    else:
        weights = np.load(os.path.join(model_dir, 'ecsw_weights_rbf_global.npy'))
    print('N_e = {}'.format(np.sum(weights > 0)))

    # Time-stepping to compute the POD-RBF PROM with ECSW
    q_snaps = U_p.T @ hdm_snaps
    t0 = time.time()
    pod_rbf_prom_q_p_snaps, man_times = inviscid_burgers_pod_rbf_2D_global_ecsw(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu_rom, U_p, U_s,
        W_global, q_p_train, q_s_train, weights, epsilon, scaler, q_snaps, kernel_type=kernel_name
    )
    
    elapsed_time = time.time() - t0
    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')

    # Reconstruct full snapshots
    num_time_steps = pod_rbf_prom_q_p_snaps.shape[1]
    pod_rbf_hprom_snaps = np.zeros((U_p.shape[0], num_time_steps))

    for i in range(num_time_steps):
        q_p_snapshot = pod_rbf_prom_q_p_snaps[:, i]
        pod_rbf_hprom_snaps[:, i] = decode_rbf_global(
            q_p_snapshot, W_global, q_p_train, U_p, U_s, epsilon, scaler, kernel_type=kernel_name
        )

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - pod_rbf_hprom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Save snapshot and/or plot
    if save_npy:
        snapshot_filename = f'pod_rbf_hprom_global_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy'
        np.save(snapshot_filename, pod_rbf_hprom_snaps)
        print(f'Snapshot saved as {snapshot_filename}')

    if save_plot:
        inds_to_plot = range(0, NUM_STEPS + 1, 100)
        snaps_to_plot = [hdm_snaps, pod_rbf_hprom_snaps]
        labels = ['HDM', 'POD-RBF HPROM (Global)']
        colors = ['black', 'green']
        linewidths = [2, 2]
        fig, ax1, ax2 = compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)
        ax1.legend(), ax2.legend()
        plot_filename = f'pod_rbf_hprom_global_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.png'
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        print(f'Plot saved as {plot_filename}')
        plt.show()

    print(f'num_its: {man_times[0]:.2f}, jac_time: {man_times[1]:.2f}, res_time: {man_times[2]:.2f}, ls_time: {man_times[3]:.2f}')

    return elapsed_time, relative_error


if __name__ == "__main__":
    main(mu1=4.56, mu2=0.019, compute_ecsw=False, save_npy=True, save_plot=True)
