import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import nnls
from joblib import Parallel, delayed

from hypernet2D import (
    load_or_compute_snaps, plot_snaps,
    inviscid_burgers_pod_gp_2D_ecsw,  
    inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
    compute_ECSW_training_matrix_2D_gp,  
    decode_gp 
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

def main(mu1=5.19, mu2=0.026, compute_ecsw=False, save_npy=False, save_plot=False):
    # Paths and parameters
    snap_folder = 'param_snaps'

    # Query point for the POD-GP PROM
    mu_rom = [mu1, mu2]

    # Load the HDM snapshots for the query parameter combination
    hdm_snaps = load_or_compute_snaps(mu_rom, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)

    ############################################################################
    # CHANGED BLOCK: Load GP model instead of global_weights.pkl
    ############################################################################
    model_dir = "pod_gp_model"  

    # Load the single multi-output GP model
    try:
        gp_models_filename = os.path.join(model_dir, 'gp_model.pkl')
        with open(gp_models_filename, 'rb') as f:
            gp_model = pickle.load(f)
        print("Single multi-output GP model loaded successfully.")
    except FileNotFoundError:
        print(f"File '{gp_models_filename}' not found.")
        exit(1)

    # Load U_p and U_s
    try:
        U_p = np.load(os.path.join(model_dir, 'U_p.npy'))
        U_s = np.load(os.path.join(model_dir, 'U_s.npy'))
        print("POD basis matrices (U_p and U_s) loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Load the Min-Max scaler for q_p
    try:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        print("Min-Max scaler for q_p loaded successfully.")
    except FileNotFoundError:
        print(f"Scaler file 'scaler.pkl' not found.")
        exit(1)
    ############################################################################

    # ECSW computation
    if compute_ecsw:
        snap_sample_factor = 10

        Clist = []
        for imu, mu in enumerate([[4.25, 0.0225]]):  # ECSW sample points
            mu_snaps = load_or_compute_snaps(mu, GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=snap_folder)
            prev_snaps = mu_snaps[:, :-1:snap_sample_factor]
            snaps = mu_snaps[:, 1::snap_sample_factor]

            print(f'Generating training block for mu = {mu}')
            Ci = compute_ECSW_training_matrix_2D_gp(
                snaps, prev_snaps, U_p, U_s, gp_model,
                inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
                GRID_X, GRID_Y, DT, mu, scaler
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
        np.save(os.path.join(model_dir,'ecsw_weights_gp.npy', weights))
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
        plt.savefig('ecsw_gp_reduced_mesh.png', dpi=300)

    else:
        weights = np.load(os.path.join(model_dir,'ecsw_weights_gp.npy'))
    print('N_e = {}'.format(np.sum(weights > 0)))

    # Time-stepping to compute the POD-GP PROM with ECSW
    t0 = time.time()
    q_snaps = U_p.T @ hdm_snaps
    # CHANGED: was inviscid_burgers_pod_rbf_2D_global_ecsw
    pod_gp_prom_q_p_snaps, man_times = inviscid_burgers_pod_gp_2D_ecsw(
        GRID_X, GRID_Y, W0, DT, NUM_STEPS, mu_rom, U_p, U_s,
        gp_model,           # CHANGED: replaces W_global
        weights,
        scaler,
        q_snaps)

    elapsed_time = time.time() - t0
    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')

    # Reconstruct full snapshots
    num_time_steps = pod_gp_prom_q_p_snaps.shape[1]
    pod_gp_hprom_snaps = np.zeros((U_p.shape[0], num_time_steps))

    for i in range(num_time_steps):
        q_p_snapshot = pod_gp_prom_q_p_snaps[:, i]
        pod_gp_hprom_snaps[:, i] = decode_gp(
            q_p_snapshot, gp_model, U_p, U_s, scaler
        )

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - pod_gp_hprom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Save snapshot and/or plot
    if save_npy:
        snapshot_filename = f'pod_gp_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy'
        np.save(snapshot_filename, pod_gp_hprom_snaps)
        print(f'Snapshot saved as {snapshot_filename}')

    if save_plot:
        inds_to_plot = range(0, NUM_STEPS + 1, 50)
        snaps_to_plot = [hdm_snaps, pod_gp_hprom_snaps]
        labels = ['HDM', 'POD-GP HPROM (Global)']
        colors = ['black', 'green']
        linewidths = [2, 2]
        fig, ax1, ax2 = compare_snaps(snaps_to_plot, inds_to_plot, labels, colors, linewidths)
        ax1.legend(), ax2.legend()
        plot_filename = f'pod_gp_hprom_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.png'
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        print(f'Plot saved as {plot_filename}')
        plt.show()

    print(f'num_its: {man_times[0]:.2f}, jac_time: {man_times[1]:.2f}, res_time: {man_times[2]:.2f}, ls_time: {man_times[3]:.2f}')

    ############################################################################
    # ADDED CODE: Create animation overlaying HDM and POD-GP using FuncAnimation
    ############################################################################
    import matplotlib.animation as animation

    fig_anim, (ax1_anim, ax2_anim) = plt.subplots(2, 1, figsize=(10, 8))

    # We define the data sets and labeling for overlay
    snaps_to_plot_anim = [hdm_snaps, pod_gp_hprom_snaps]
    labels_anim = ['HDM', 'POD-GP HPROM']
    colors_anim = ['black', 'green']
    linewidths_anim = [2, 2]

    def animate_func(frame_idx):
        ax1_anim.clear()
        ax2_anim.clear()

        # Fix the y-limits for both subplots
        ax1_anim.set_ylim(0, 6.5)
        ax2_anim.set_ylim(0, 6.5)

        # Overlay both HDM & POD-GP for this single time index 'frame_idx'
        for i, each_snaps in enumerate(snaps_to_plot_anim):
            plot_snaps(
                GRID_X, GRID_Y,
                each_snaps, [frame_idx],
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
            frames=range(num_time_steps),
            interval=300,  # ms between frames
            blit=False,
            repeat=False
        )

        anim_filename = f'pod_gp_hprom_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.gif'
        anim.save(anim_filename, writer='imagemagick', fps=30)
        print(f"Saved animation '{anim_filename}' with overlay of HDM & POD-GP at each timestep.")

    return elapsed_time, relative_error


if __name__ == "__main__":
    main(mu1=4.75, mu2=0.02, compute_ecsw=True, save_npy=False, save_plot=True)
