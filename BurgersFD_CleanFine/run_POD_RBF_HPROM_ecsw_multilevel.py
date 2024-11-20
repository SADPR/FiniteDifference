import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

from scipy.optimize import nnls

from hypernet2D import (load_or_compute_snaps, make_2D_grid,
                        plot_snaps, inviscid_burgers_pod_rbf_2D_ecsw,
                        inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
                        compute_ECSW_training_matrix_2D_rbf, decode_rbf)
from config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU

plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=13)

def compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths):
  fig, (ax1, ax2) = plt.subplots(2, 1)
  for i, snaps in enumerate(snaps_to_plot):
    plot_snaps(grid_x, grid_y, snaps, inds_to_plot,
               label=labels[i],
               fig_ax=(fig, ax1, ax2),
               color=colors[i],
               linewidth=linewidths[i])

  return fig, ax1, ax2

def main(mu1=4.75, mu2=0.02, compute_ecsw=False):
    # Paths and parameters
    snap_folder = 'param_snaps'

    # Query point for the POD-RBF PROM
    mu_rom = [mu1, mu2]

    # Sample points for ECSW
    mu_samples = [[4.25, 0.0225]]  # You can adjust this as needed

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 750, 750
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x))
    v0 = np.ones((num_cells_y, num_cells_x))
    w0 = np.concatenate((u0.flatten(), v0.flatten()))

    # Load the HDM snapshots for the query parameter combination
    hdm_snaps = load_or_compute_snaps(mu_rom, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)

    # Load the KDTree and training data for RBF interpolation
    with open('modes/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
        kdtree = data['KDTree']
        q_p_train = data['q_p']
        q_s_train = data['q_s']
    
    # Load the scaler
    with open('modes/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the POD basis matrices
    U_p = np.load('modes/U_p.npy')
    U_s = np.load('modes/U_s.npy')

    # Set epsilon and neighbors based on the value of mu1 (adjust as needed)
    if mu1 == 4.75:
        epsilon = 0.01
        neighbors = 22
    elif mu1 == 4.56:
        epsilon = 0.01
        neighbors = 25
    elif mu1 == 5.19:
        epsilon = 0.01
        neighbors = 25
    else:
        raise ValueError(f"Unsupported mu1 value: {mu1}")

    print(f"mu1: {mu1}, epsilon: {epsilon}, neighbors: {neighbors}")

    # ECSW
    if compute_ecsw:
        snap_sample_factor = 10

        Clist = []
        for imu, mu in enumerate(mu_samples):
            mu_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
            prev_snaps = mu_snaps[:, :-1:snap_sample_factor]
            snaps = mu_snaps[:, 1::snap_sample_factor]
            n_snaps = snaps.shape[1]

            print('Generating training block for mu = {}'.format(mu))
            
            Ci = compute_ECSW_training_matrix_2D_rbf(snaps, prev_snaps, U_p, U_s, epsilon, neighbors,
                                        kdtree, q_p_train, q_s_train, inviscid_burgers_res2D, inviscid_burgers_exact_jac2D, grid_x, grid_y, dt, mu, scaler, kernel_type='gaussian')
            Clist.append(Ci)

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
        np.save('ecsw_weights_rbf_multilevel', weights)
        '''
        plt.clf()
        plt.rc('font', size=16)
        plt.spy(weights.reshape((num_cells_y, num_cells_x)))
        plt.xlabel('$x$ cell index')
        plt.ylabel('$y$ cell index')
        plt.title('POD-RBF Reduced Mesh')
        plt.tight_layout()
        plt.savefig('pod-rbf-reduced-mesh.png', dpi=300)
        plt.show()
        '''
    else:
        weights = np.load('ecsw_weights_rbf_multilevel.npy')
    print('N_e = {}'.format(np.sum(weights > 0)))
    # END ECSW

    # Time-stepping to compute the POD-RBF PROM at the out-of-sample parameter point
    t0 = time.time()
    pod_rbf_prom_q_p_snaps, man_times = inviscid_burgers_pod_rbf_2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu_rom, U_p, U_s,
                                     epsilon, neighbors, kdtree, q_p_train, q_s_train, weights, scaler, kernel_type='gaussian')
    man_its, man_jac, man_res, man_ls = man_times
    elapsed_time = time.time() - t0
    print(f'Elapsed ROM time: {elapsed_time:.3e} seconds')

    # Initialize the matrix to store reconstructed snapshots
    num_time_steps = pod_rbf_prom_q_p_snaps.shape[1]
    pod_rbf_hprom_snaps = np.zeros((U_p.shape[0], num_time_steps))

    # Reconstruct each snapshot
    for i in range(num_time_steps):
        # Decode each primary coordinate vector to full state
        q_p_snapshot = pod_rbf_prom_q_p_snaps[:, i]
        pod_rbf_hprom_snaps[:, i] = decode_rbf(q_p_snapshot, epsilon, neighbors, kdtree, q_p_train, q_s_train, U_p, U_s, scaler, kernel_type="gaussian")

    # Calculate relative error
    relative_error = 100 * np.linalg.norm(hdm_snaps - pod_rbf_hprom_snaps) / np.linalg.norm(hdm_snaps)
    print(f'Relative error: {relative_error:.2f}%')

    # Save the snapshot to a file
    np.save(f'dd_multilevel_pod_rbf_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', pod_rbf_hprom_snaps)
    print(f'Snapshot saved as pod_rbf_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy')

    # Optionally plot and compare snapshots
    
    inds_to_plot = range(0, num_steps + 1, 100)
    snaps_to_plot = [hdm_snaps, pod_rbf_hprom_snaps]
    labels = ['HDM', 'POD-RBF']
    colors = ['black', 'green']
    linewidths = [2, 1]
    fig, ax1, ax2 = compare_snaps(grid_x, grid_y, snaps_to_plot, inds_to_plot, labels, colors, linewidths)

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    save_path = f'dd_multilevel_hprom_pod-rbf_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.png'
    print(f'Saving as "{save_path}"')
    plt.savefig(save_path, dpi=300)
    plt.show()
    

    # Print timings for the steps
    print(f'num_its: {man_its:.2f}, jac_time: {man_jac:.2f}, res_time: {man_res:.2f}, ls_time: {man_ls:.2f}')

    # Return elapsed time and relative error
    return elapsed_time, relative_error

if __name__ == "__main__":
    main(mu1=4.75, mu2= 0.02, compute_ecsw=True)
