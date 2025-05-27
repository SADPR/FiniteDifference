import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.optimize import nnls

from hypernet2D import (load_or_compute_snaps, make_2D_grid,
                        plot_snaps, inviscid_burgers_pod_rbf_2D_ecsw,
                        inviscid_burgers_res2D, inviscid_burgers_exact_jac2D,
                        compute_ECSW_training_matrix_2D_rbf, decode_rbf)
from config import MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU
from empirical_cubature_method import EmpiricalCubatureMethod
from randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

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

def main(mu1=5.19, mu2=0.026, compute_ecsw=False):
    # Paths and parameters
    snap_folder = 'param_snaps'

    # Query point for the POD-RBF PROM
    mu_rom = [mu1, mu2]

    # Sample points for ECSW
    mu_samples = [[4.25, 0.0225]]  # You can adjust this as needed

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
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
            

            '''
            snap_sample_factor_1 = 10
            snap_sample_factor_2 = 10
            mu_snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
            # Select every 10 snapshots from the start to index 400
            prev_snaps_part1 = mu_snaps[:, :470:snap_sample_factor_1]
            snaps_part1 = mu_snaps[:, 1:470:snap_sample_factor_1]
            
            # Select every 2 snapshots from index 400 to the end
            prev_snaps_part2 = mu_snaps[:, 470:-1:snap_sample_factor_2]
            snaps_part2 = mu_snaps[:, 471::snap_sample_factor_2]
            
            # Concatenate both parts along the snapshot axis
            prev_snaps = np.concatenate((prev_snaps_part1, prev_snaps_part2), axis=1)
            snaps = np.concatenate((snaps_part1, snaps_part2), axis=1)
            
            n_snaps = snaps.shape[1]
            '''

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
        selected_indices = (idxs == 1).ravel()
        C = C[:, selected_indices]

        # Weighting for boundary
        bc_w = 10

        t1 = time.time()
        C = np.ascontiguousarray(C, dtype=np.float64)
        b = np.ascontiguousarray(C.sum(axis=1), dtype=np.float64)
        u,_,_,_= RandomizedSingularValueDecomposition().Calculate(C.T, 1e-6)
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

        weights = weights.reshape((num_cells_y - 2 * nn_y, num_cells_x - 2 * nn_x))
        full_weights = bc_w * np.ones((num_cells_y, num_cells_x))
        full_weights[idxs > 0] = weights.ravel()
        weights = full_weights.ravel()
        np.save('ecsw_weights_rbf', weights)
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
        weights = np.load('ecsw_weights_rbf.npy')
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
    np.save(f'pod_rbf_hprom_snaps_mu1_{mu_rom[0]:.2f}_mu2_{mu_rom[1]:.3f}.npy', pod_rbf_hprom_snaps)
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
    save_path = f'pod-rbf_{mu_rom[0]:.2f}_{mu_rom[1]:.3f}.png'
    print(f'Saving as "{save_path}"')
    plt.savefig(save_path, dpi=300)
    plt.show()
    

    # Print timings for the steps
    print(f'num_its: {man_its:.2f}, jac_time: {man_jac:.2f}, res_time: {man_res:.2f}, ls_time: {man_ls:.2f}')

    # Return elapsed time and relative error
    return elapsed_time, relative_error

if __name__ == "__main__":
    main(mu1=4.75, mu2= 0.02, compute_ecsw=False)
