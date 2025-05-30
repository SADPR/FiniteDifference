"""
Use the Burgers equation to try out some learning-based hyper-reduction approaches
"""

import glob
import time

import numpy as np
from numpy.linalg import norm
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch
import functorch
import os
from sklearn.utils.extmath import randomized_svd
from scipy.spatial.distance import pdist, squareform
import pickle
from rbf_utils import RBFUtils

COUNTER = 0


plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "stix",
    "font.family": ["STIXGeneral"]})
plt.rc('font', size=16)

def inviscid_burgers_explicit2D(grid_x, grid_y, u0, v0, dt, num_steps, mu):
    """
    """

    snaps = np.zeros((u0.flatten().size + v0.flatten().size, num_steps+1))
    snaps[:, 0] = np.concatenate((u0.flatten(), v0.flatten()))
    up = u0.copy()
    vp = v0.copy()
    dx = grid_x[1:] - grid_x[:-1]
    xc = (grid_x[1:] + grid_x[:-1])/2

    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    b = np.zeros_like(up)
    b[:, 0] = 0.5 * mu[0]**2
    by = np.zeros_like(vp)
    by[0, :] = 0.5 * mu[0]**2
    f = np.zeros(grid_x.size)
    f[0] = 0.5 * mu[0]**2
    for i in range(num_steps):
        Fux = (0.5 * np.square(up)).T
        Fvy = 0.5 * np.square(vp)
        Fuv = 0.5 * up*vp
        FuvT = Fuv.T
        u = up - dt * ((Dxec@Fux).T - b/dx) + dt*0.02*np.exp(mu[1]*xc[None, :]) \
            - dt * Dyec @ Fuv
        v = vp - dt * Dyec@Fvy\
            - dt * (Dxec@FuvT).T
        if i % 10 == 0:
            print('... Working on timesetp {}'.format(i))
        if i % 200 == 0:
            plt.imshow(v)
            plt.colorbar()
            plt.title('i = {}'.format(i))
            plt.show()
            time.sleep(0.2)
        if i in range(499, 5001, 500):
            snaps[:, i + 1] = np.concatenate((u.ravel(), v.ravel()))
        up = u
        vp = v
    return snaps

def inviscid_burgers_implicit2D(grid_x, grid_y, w0, dt, num_steps, mu):
    """
    Solves the 2D inviscid Burgers' equation using implicit time-stepping and Newton-Raphson.
    
    Parameters:
    - grid_x, grid_y: Discretized grid points in x and y directions.
    - w0: Initial conditions (for both u and v components).
    - dt: Time step size.
    - num_steps: Number of time steps to solve.
    - mu: Parameter vector controlling boundary conditions and source term.
    
    Returns:
    - snaps: Snapshots of the solution at each time step.
    """

    # Initialization
    print("Running HDM for mu1={}".format(mu[0]))
    snaps = np.zeros((w0.size, num_steps + 1))  # Array to store snapshots of the solution
    snaps[:, 0] = w0.ravel().copy()  # Store the initial condition
    wp = w0.ravel()  # Previous time step solution

    # Create derivative operators for x and y directions (first-order difference matrices)
    Dxec = make_ddx(grid_x)  # Derivative matrix in x direction
    Dyec = make_ddx(grid_y)  # Derivative matrix in y direction

    # Create block derivative operators (kron product) for the full system
    JDxec = sp.kron(sp.eye(grid_y.size - 1), Dxec)  # Block derivative for x
    JDyec = sp.kron(sp.eye(grid_x.size - 1), Dyec)  # Block derivative for y

    # Adjust for indexing in the y direction (reordering)
    JDyec = JDyec.tocsr()
    idx = np.arange((grid_y.size - 1) * (grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)).T.ravel()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]

    # Identity matrix for the full system (for implicit time-stepping)
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))

    # Loop over time steps
    for i in range(num_steps):
        # Define the residual function using alternative formulation (for nonlinearity)
        def res(w):
            return inviscid_burgers_res2D_alt(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec)

        # Define the Jacobian function (linearization of the residual)
        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        # Solve the nonlinear system using Newton-Raphson iteration
        print(" ... Working on timestep {}".format(i))
        w, resnorms = newton_raphson(res, jac, wp, max_its=100, relnorm_cutoff=1e-12)

        # Store the solution for the current time step
        snaps[:, i + 1] = w.ravel()

        # Update the previous solution for the next time step
        wp = w.copy()

    return snaps

def inviscid_burgers_implicit2D_LSPG(grid_x, grid_y, w0, dt, num_steps, mu, basis):
    """
    Solve the 2D inviscid Burgers' equation using LSPG (Least-Squares Petrov-Galerkin) with 
    an implicit time integrator and a reduced-order model (ROM).

    Parameters:
    - grid_x, grid_y: Grids in the x and y directions.
    - w0           : Initial condition vector.
    - dt           : Time step size.
    - num_steps    : Number of time steps.
    - mu           : Parameter vector [mu1, mu2].
    - basis        : POD basis for the reduced-order model (ROM).

    Returns:
    - snaps        : Snapshots of the reduced-order solutions.
    - (num_its, jac_time, res_time, ls_time): Metrics for iterations, Jacobian, residuals, and least-squares times.
    """

    # Initialize counters and timing metrics
    num_its, jac_time, res_time, ls_time = 0, 0, 0, 0
    npod = basis.shape[1]  # Number of POD modes
    snaps = np.zeros((w0.size, num_steps+1))  # Storage for snapshots
    red_coords = np.zeros((npod, num_steps+1))  # Storage for reduced coordinates
    
    # Project initial condition onto the reduced basis
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:, 0], red_coords[:, 0] = w0, y0
    wp, yp = w0.copy(), y0.copy()

    # Precompute derivative matrices and identity matrix for system size
    Dxec, Dyec = make_ddx(grid_x), make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()[np.arange((grid_y.size - 1)**2).reshape((grid_y.size - 1), -1).T.flatten(), :]
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))

    print(f"Running ROM of size {npod} for mu1={mu[0]}, mu2={mu[1]}")

    # Residual and Jacobian functions
    def compute_residual(w):
        """Compute the residual for the current timestep."""
        return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

    def compute_jacobian(w):
        """Compute the Jacobian matrix for the current timestep."""
        return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

    # Time-stepping loop
    for i in range(num_steps):
        print(f" ... Working on timestep {i}")

        # Solve for reduced coordinates using Gauss-Newton method
        y, resnorms, times = gauss_newton_LSPG(compute_residual, compute_jacobian, basis, yp)
        jac_timep, res_timep, ls_timep = times
        
        # Update iteration counts and timings
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        # Reconstruct the full solution from reduced coordinates
        w = basis.dot(y)
        red_coords[:, i+1], snaps[:, i+1] = y.copy(), w.copy()
        wp, yp = w.copy(), y.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_ecsw_fixed(grid_x, grid_y, weights, w0, dt, num_steps, mu, basis):
    """
    """

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    npod = basis.shape[1]
    snaps = np.zeros((w0.size, num_steps + 1))
    red_coords = np.zeros((npod, num_steps + 1))
    y0 = basis.T.dot(w0)
    w0 = basis.dot(y0)
    snaps[:, 0] = w0
    red_coords[:, 0] = y0
    wp = w0.copy()
    yp = y0.copy()

    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]

    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size/2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()
    print("Running ROM of size {} for mu1={}, mu2={}".format(npod, mu[0], mu[1]))

    weights2 = np.hstack((weights, weights))
    sample_weights = weights2[sample_inds]

    idx = np.concatenate((augmented_sample, int(w0.size/2) + augmented_sample))
    wp = w0[idx]

    basis_red = basis[idx, :]
    for i in range(num_steps):

        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_ECSW_2D(res, jac, basis_red, yp, sample_inds, augmented_sample, sample_weights)
        print('number iter: {}'.format(len(resnorms)))
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w = basis_red.dot(y)
        # u, v = np.split(w, 2)
        # plt.imshow(u.reshape(250, 250))
        # plt.colorbar()
        # plt.show()

        red_coords[:, i + 1] = y.copy()
        wp = w.copy()
        yp = y.copy()

    return red_coords, (jac_time, res_time, ls_time)

import numpy as np
import scipy.sparse as sp
import torch

def inviscid_burgers_implicit2D_ae_LSPG(grid_x, grid_y, w0, dt, num_steps, mu, autoencoder):
    """
    Solve the 2D inviscid Burgers' equation using LSPG with an autoencoder-based ROM.

    Parameters:
    - grid_x, grid_y: Grids in the x and y directions.
    - w0           : Initial condition vector.
    - dt           : Time step size.
    - num_steps    : Number of time steps.
    - mu           : Parameter vector [mu1, mu2].
    - autoencoder  : Trained autoencoder model (PyTorch).

    Returns:
    - snaps        : Snapshots of the reduced-order solutions.
    - (num_its, jac_time, res_time, ls_time): Metrics for iterations.
    """

    num_its, jac_time, res_time, ls_time = 0, 0, 0, 0
    latent_dim = autoencoder.encoder[0].out_features  # Extract latent space size
    snaps = np.zeros((w0.size, num_steps+1))
    red_coords = np.zeros((latent_dim, num_steps+1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    autoencoder.eval()

    # Convert w0 to PyTorch tensor and encode
    w0_tensor = torch.tensor(w0, dtype=torch.float32).to(device)
    z0 = autoencoder.encoder(w0_tensor).detach().cpu().numpy()
    w0_recon = autoencoder.decoder(torch.tensor(z0, dtype=torch.float32).to(device)).detach().cpu().numpy()

    snaps[:, 0], red_coords[:, 0] = w0_recon, z0
    wp, zp = w0_recon.copy(), z0.copy()

    Dxec, Dyec = make_ddx(grid_x), make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()[np.arange((grid_y.size - 1)**2).reshape((grid_y.size - 1), -1).T.flatten(), :]
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))

    print(f"Running ROM of size {latent_dim} for mu1={mu[0]}, mu2={mu[1]}")

    def compute_residual(w):
        return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

    def compute_jacobian(w):
        return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

    for i in range(num_steps):
        print(f" ... Working on timestep {i}")

        # Solve for reduced coordinates using Gauss-Newton method
        z, resnorms, times = gauss_newton_ae_LSPG(compute_residual, compute_jacobian, autoencoder.decoder, zp)
        jac_timep, res_timep, ls_timep = times

        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        w_recon = autoencoder.decoder(torch.tensor(z, dtype=torch.float32).to(device)).detach().cpu().numpy()

        red_coords[:, i+1], snaps[:, i+1] = z.copy(), w_recon.copy()
        wp, zp = w_recon.copy(), z.copy()

    return snaps, (num_its, jac_time, res_time, ls_time)


def inviscid_burgers_rnm2D(grid_x, grid_y, w0, dt, num_steps, mu, rnm, ref, basis, basis2):
    """
    Solves the 2D inviscid Burgers' equations using a Reduced-Order Model (ROM) 
    augmented with Proper Orthogonal Decomposition (POD) and an Artificial Neural Network (ANN).
    
    Utilizes a first-order Godunov spatial discretization and a second-order trapezoidal
    rule time integrator within the LSPG (Least-Squares Petrov-Galerkin) framework.
    
    Parameters:
    - grid_x, grid_y: Arrays defining the grid points in the x and y directions.
    - w0: Initial state vector (flattened for both u and v components).
    - dt: Time step size.
    - num_steps: Number of time steps to simulate.
    - mu: Parameter vector [mu1, mu2], where:
        mu[0]: Inlet state value.
        mu[1]: Exponential rate of the source term.
    - rnm: Trained ANN model representing the nonlinear map \(\mathcal{N}(\mathbf{q})\).
    - ref: Reference solution vector (not directly used in this snippet).
    - basis: Primary POD modes matrix (\(\mathbf{V}\)).
    - basis2: Secondary POD modes matrix (\(\mathbf{\bar{V}}\)).
    
    Returns:
    - snaps: Array of solution snapshots at each time step.
    - (num_its, jac_time, res_time, ls_time): Tuple containing counts of iterations and timing metrics.
    """
    
    # -----------------------------------
    # 1. Operators Setup
    # -----------------------------------
    
    # Create finite difference derivative operators for x and y directions
    Dxec = make_ddx(grid_x)  # Derivative operator in x-direction
    Dyec = make_ddx(grid_y)  # Derivative operator in y-direction
    
    # Construct full Jacobian derivative matrices using Kronecker products
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)  # For u-component
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)  # For v-component
    
    # Convert Jacobian matrices to Compressed Sparse Row format for efficiency
    JDyec = JDyec.tocsr()
    
    # Reshape indices to match the grid layout
    shp = (grid_y.size - 1, grid_y.size - 1)
    size = shp[0] * shp[1]
    idx = np.arange(size).reshape(shp).T.flatten()
    
    # Reorder Jacobian matrix rows and columns to align with the grid
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    
    # Create an identity matrix for stability and time integration contributions
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))
    
    # Assemble the block Jacobian operator for both u and v components
    Jop = sp.bmat([[JDxec, None], [None, JDyec]])
    
    # -----------------------------------
    # 2. Initialize Counters for Performance Metrics
    # -----------------------------------
    
    num_its = 0      # Total number of iterations across all time steps
    jac_time = 0     # Total time spent computing Jacobians
    res_time = 0     # Total time spent computing residuals
    ls_time = 0      # Total time spent in least-squares solves
    
    # -----------------------------------
    # 3. Prepare Initial State
    # -----------------------------------
    
    # Convert parameters to a PyTorch tensor for potential use in the ANN
    tmu = torch.tensor(mu, dtype=torch.float)
    
    # Flatten and reshape the initial state vector for u and v components
    w0 = torch.tensor(w0.ravel(), dtype=torch.float).unsqueeze(0).unsqueeze(0)
    
    # Project the initial full state onto the primary POD basis to obtain reduced coordinates
    y0 = basis.T @ w0.squeeze()
    
    with torch.no_grad():
        # Reconstruct the full state using POD and ANN without tracking gradients
        # Here, rnm(y0) represents the ANN's output for the nonlinear mapping
        w0 = basis @ y0 + basis2 @ rnm(y0)  # \(\tilde{\mathbf{u}} = \mathbf{V}\mathbf{q} + \mathbf{\bar{V}}\mathcal{N}(\mathbf{q})\)
    
    # Determine the size of the reduced coordinate vector
    nred = y0.shape[0]
    
    # Initialize arrays to store solution snapshots and reduced coordinates over time
    snaps = np.zeros((w0.shape[0], num_steps + 1))        # Full state snapshots
    red_coords = np.zeros((nred, num_steps + 1))         # Reduced coordinates
    snaps[:, 0] = w0.squeeze().numpy()                    # Set initial snapshot
    red_coords[:, 0] = y0.squeeze().numpy()              # Set initial reduced coordinates
    
    # Clone the initial state for iterative updates
    wp = w0.detach().clone()  # Previous state (w_p)
    yp = y0.detach().clone()  # Previous reduced coordinates (y_p)
    
    # -----------------------------------
    # 4. Define the Reconstruction Function
    # -----------------------------------
    
    def decode(x, with_grad=True):
        """
        Reconstruct the full state vector from reduced coordinates using POD and ANN.
        
        Parameters:
        - x: Reduced coordinates (\(\mathbf{q}\)).
        - with_grad: If True, enables gradient tracking for Jacobian computation.
        
        Returns:
        - Reconstructed full state vector (\(\tilde{\mathbf{u}}\)).
        """
        if with_grad:
            # Include gradient tracking for derivative calculations
            return basis @ x + basis2 @ rnm(x)
        else:
            # Reconstruct without tracking gradients for efficiency
            with torch.no_grad():
                return basis @ x + basis2 @ rnm(x)
    
    # Compute the Jacobian of the decode function with respect to reduced coordinates
    jacfwdfunc = functorch.jacfwd(decode)
    
    # -----------------------------------
    # 5. Benchmarking Jacobian and ANN Evaluation Times (Optional)
    # -----------------------------------
    
    # Measure average time to compute one Jacobian using automatic differentiation
    t0 = time.time()
    for i in range(100):
        jacfwdfunc(yp)
    print('Time to compute 1 jacobian: {:3.2e}'.format((time.time() - t0)/100))
    
    # Measure average time to evaluate the ANN
    t0 = time.time()
    for i in range(100):
        rnm(yp)  # Evaluate ANN with current reduced coordinates
    print('Time to evaluate network: {:3.2e}'.format((time.time() - t0)/100))
    
    # -----------------------------------
    # 6. Time-Stepping Loop for ROM Simulation
    # -----------------------------------
    
    print("Running M-ROM of size {} for mu1={}, mu2={}".format(nred, mu[0], mu[1]))
    
    for i in range(num_steps):
        # Define the residual function for the current state
        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp.squeeze().numpy(), mu, Dxec, Dyec)
    
        # Define the Jacobian function for the current state
        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)
    
        print(" ... Working on timestep {}".format(i))
        
        t0 = time.time()
        # Perform Gauss-Newton iterations to solve for updated reduced coordinates
        y, resnorms, times = gauss_newton_rnm(res, jac, yp, decode, jacfwdfunc)
        print(f"Time to gauss newton: {time.time() - t0:.6f} seconds")
        
        # Unpack timing metrics from the solver
        jac_timep, res_timep, ls_timep = times
        
        # Accumulate performance metrics
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep
    
        with torch.no_grad():
            # Reconstruct the full state using the updated reduced coordinates
            # \(\tilde{\mathbf{u}} = \mathbf{V}\mathbf{q} + \mathbf{\bar{V}}\mathcal{N}(\mathbf{q})\)
            w = basis @ y + basis2 @ rnm(y)
    
        # Store the updated reduced coordinates and full state snapshot
        red_coords[:, i + 1] = y.squeeze().numpy()
        snaps[:, i + 1] = w.squeeze().numpy()
        
        # Update previous state vectors for the next iteration
        wp = w.detach().clone()
        yp = y.detach().clone()
    
    # Return all solution snapshots and performance metrics
    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_rnm2D_joshua(grid_x, grid_y, w0, dt, num_steps, mu, rnm, ref, basis, basis2):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 1D burgers
    problem with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    # Operators setup
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    shp = (grid_y.size - 1, grid_y.size - 1)
    size = shp[0] * shp[1]
    idx = np.arange(size).reshape(shp).T.flatten()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    Eye = sp.eye(2 * (grid_x.size - 1) * (grid_y.size - 1))
    Jop = sp.bmat([[JDxec, None], [None, JDyec]])

    # Initialize counters
    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Prepare initial state
    tmu = torch.tensor(mu, dtype=torch.float)
    w0 = torch.tensor(w0.ravel(), dtype=torch.float).unsqueeze(0).unsqueeze(0)
    y0 = basis.T @ w0.squeeze()
    
    with torch.no_grad():
        # **Removed concatenation of y0 and tmu**
        w0 = basis @ y0 + basis2 @ rnm(y0)  # Updated to pass only y0

    nred = y0.shape[0]
    snaps = np.zeros((w0.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0.squeeze().numpy()
    red_coords[:, 0] = y0.squeeze().numpy()
    wp = w0.detach().clone()
    yp = y0.detach().clone()

    def decode(x, with_grad=True):
        if with_grad:
            # **Removed concatenation of x and tmu**
            return basis @ x + basis2 @ rnm(x)  # Updated to pass only x
        else:
            with torch.no_grad():
                return basis @ x + basis2 @ rnm(x)  # Updated to pass only x

    jacfwdfunc = functorch.jacfwd(decode)
    t0 = time.time()
    for i in range(100):
        jacfwdfunc(yp)
    print('Time to compute 1 jacobian: {:3.2e}'.format((time.time() - t0)/100))
    
    t0 = time.time()
    for i in range(100):
        # **Removed concatenation of yp and tmu**
        rnm(yp)  # Updated to pass only yp
    print('Time to evaluate network: {:3.2e}'.format((time.time() - t0)/100))

    print("Running M-ROM of size {} for mu1={}, mu2={}".format(nred, mu[0], mu[1]))
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp.squeeze().numpy(), mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(" ... Working on timestep {}".format(i))
        y, resnorms, times = gauss_newton_rnm(res, jac, yp, decode, jacfwdfunc)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        with torch.no_grad():
            # **Removed concatenation of y and tmu**
            w = basis @ y + basis2 @ rnm(y)  # Updated to pass only y

        red_coords[:, i + 1] = y.squeeze().numpy()
        snaps[:, i + 1] = w.squeeze().numpy()
        wp = w.detach().clone()
        yp = y.detach().clone()

    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_rnm2D_ecsw(grid_x, grid_y, w0, dt, num_steps, mu, rnm, ref, basis, basis2, weights):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 1D burgers
    problem with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term

    so the equation solved is
    w_t + (0.5 * w^2)_x = 0.02 * exp(mu[1]*x)
    w(x=grid[0], t) = mu[0]
    w(x, t=0) = w0
    """

    # stuff for operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]

    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size / 2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()

    sample_weights = np.concatenate((weights, weights))[sample_inds]

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0
    tmu = torch.tensor(mu.copy(), dtype=torch.float)
    w0 = torch.tensor(w0.copy().ravel(), dtype=torch.float).unsqueeze(0).unsqueeze(0)
    y0 = basis.T@w0.squeeze()
    with torch.no_grad():
        w0 = basis@y0 + basis2@rnm(torch.cat((y0, tmu))) #basis2@rnm(y0)
    nred = y0.shape[0]
    snaps = np.zeros((w0.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0.squeeze().numpy()
    red_coords[:, 0] = y0.squeeze().numpy()
    wp = w0.detach().clone()
    yp = y0.detach().clone()

    idx = np.concatenate((augmented_sample, int(w0.shape[0] / 2) + augmented_sample))
    wp = w0[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]
    def decode(x, with_grad=True):
        if with_grad:
            return V @ x + Vbar @ rnm(torch.cat((x, tmu))) #Vbar@rnm(x)
        else:
            with torch.no_grad():
                return V @ x + Vbar @ rnm(torch.cat((x, tmu))) #Vbar@rnm(x)

    jacfwdfunc = functorch.jacfwd(decode)
    t0 = time.time()
    for i in range(100):
        jacfwdfunc(yp)
    print('Time to compute 1 jacobian: {:3.2e}'.format((time.time() - t0)/100))
    t0 = time.time()
    for i in range(100):
        rnm(torch.cat((yp, tmu)))#rnm(yp)
    print('Time to evaluate network: {:3.2e}'.format((time.time() - t0)/100))


    print("Running M-ROM of size {} for mu1={}, mu2={}".format(nred, mu[0], mu[1]))
    lbc = None
    src = None
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    wall_clock_time = 0.0
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc, src)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(" ... Working on timestep {}".format(i))
        t0 = time.time()
        y, resnorms, times = gauss_newton_rnm_ecsw(res, jac, yp, decode, jacfwdfunc, sample_inds, augmented_sample, sample_weights)
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        with torch.no_grad():
            w = V @ y + Vbar @ rnm(torch.cat((y, tmu))) #Vbar@rnm(y)

        red_coords[:, i + 1] = y.squeeze().detach().numpy()
        wp = w.detach().clone()
        yp = y.detach().clone()
        wall_clock_time += (time.time() - t0)

    return red_coords, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_pod_rbf_2D_nearest_neighbors(grid_x, grid_y, w0, dt, num_steps, mu, basis, basis2,
                                                  epsilon, neighbors, kdtree, q_p_train, q_s_train, scaler, kernel_type="gaussian"):
    """
    Solves the 2D inviscid Burgers' equations using a Reduced-Order Model (ROM)
    augmented with Proper Orthogonal Decomposition (POD) and Radial Basis Functions (RBF).
    
    Parameters:
    - grid_x, grid_y: Arrays defining the grid points in the x and y directions.
    - w0: Initial state vector (flattened for both u and v components).
    - dt: Time step size.
    - num_steps: Number of time steps to simulate.
    - mu: Parameter vector [mu1, mu2].
    - basis: Primary POD modes matrix (\(\mathbf{V}\)).
    - basis2: Secondary POD modes matrix (\(\mathbf{\bar{V}}\)).
    - epsilon: Shape parameter for RBF.
    - neighbors: Number of neighbors for RBF interpolation.
    - kdtree: KDTree used for nearest-neighbor search.
    - q_p_train, q_s_train: Training data for RBF interpolation.
    - scaler: Scaler for normalization (e.g., MinMaxScaler or StandardScaler).
    - kernel_type: Type of RBF kernel to use (e.g., "gaussian").
    
    Returns:
    - snaps: Array of solution snapshots at each time step.
    - (num_its, jac_time, res_time, ls_time): Performance metrics.
    """

    # -----------------------------------
    # 1. Operators Setup
    # -----------------------------------
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initial conditions
    y0 = basis.T @ w0  # Project w0 onto the POD basis
    w0_reconstructed = decode_rbf_nearest_neighbors(y0, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)

    nred = y0.shape[0]
    snaps = np.zeros((w0_reconstructed.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0_reconstructed
    red_coords[:, 0] = y0

    wp = w0_reconstructed.copy()
    yp = y0.copy()

    # Decode function using POD-RBF
    def decode_func(x):
        return decode_rbf_nearest_neighbors(x, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)

    # Jacobian function for POD-RBF
    def jac_rbf_func(x):
        return jac_rbf_nearest_neighbors(x, kdtree, q_p_train, q_s_train, basis, basis2, epsilon, neighbors, scaler, kernel_type)

    print(f"Running POD-RBF Nearest Neighbors of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    # Time-Stepping Loop
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(f" ... Working on timestep {i}")
        t0 = time.time()

        # Solve using Gauss-Newton for POD-RBF
        y, resnorms, times = gauss_newton_pod_rbf(
            res, jac, yp, decode_func, jac_rbf_func
        )
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        # Reconstruct the full state
        w_reconstructed = decode_func(y)

        red_coords[:, i + 1] = y
        snaps[:, i + 1] = w_reconstructed
        wp = w_reconstructed
        yp = y

    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_pod_rbf_2D_nearest_neighbors_ecsw(grid_x, grid_y, w0, dt, num_steps, mu, basis, basis2,
                                     epsilon, neighbors, kdtree, q_p_train, q_s_train, weights, scaler, kernel_type='gaussian'):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 1D burgers
    problem with a source term. The parameters are as follows:
    mu[0]: inlet state value
    mu[1]: the exponential rate of the exponential source term
    """
    
    # stuff for operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    # Mesh sampling based on ECSW weights
    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size / 2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()

    sample_weights = np.concatenate((weights, weights))[sample_inds]

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initial conditions
    y0 = basis.T @ w0  # Project w0 onto the POD basis
    w0_reconstructed = decode_rbf_nearest_neighbors(y0, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)

    nred = y0.shape[0]
    snaps = np.zeros((w0_reconstructed.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0_reconstructed
    red_coords[:, 0] = y0

    wp = w0_reconstructed.copy()
    yp = y0.copy()

    # Reduced basis for sampled nodes
    idx = np.concatenate((augmented_sample, int(w0_reconstructed.shape[0] / 2) + augmented_sample))
    wp = wp[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]

    # Decode function using POD-RBF
    def decode_func(x):
        return decode_rbf_nearest_neighbors(x, epsilon, neighbors, kdtree, q_p_train, q_s_train, V, Vbar, scaler, kernel_type)

    # Jacobian function for POD-RBF
    def jac_rbf_func(x):
        return jac_rbf_nearest_neighbors(x, kdtree, q_p_train, q_s_train, V, Vbar, epsilon, neighbors, scaler, kernel_type)

    print(f"Running POD-RBF M-ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")
    lbc = None
    src = None
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)

    # Initialize boundary conditions and source terms
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    wall_clock_time = 0.0

    # Time-stepping loop
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc, src)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(f" ... Working on timestep {i}")
        t0 = time.time()

        # Solve using Gauss-Newton for POD-RBF
        y, resnorms, times = gauss_newton_pod_rbf_ecsw(
            res, jac, yp, decode_func, jac_rbf_func, sample_inds, augmented_sample, sample_weights
        )
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        # Reconstruct the full state
        w_reconstructed = decode_rbf_nearest_neighbors(y, epsilon, neighbors, kdtree, q_p_train, q_s_train, V, Vbar, scaler, kernel_type)

        red_coords[:, i + 1] = y
        wp = w_reconstructed
        yp = y
        wall_clock_time += time.time() - t0

    return red_coords, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_pod_rbf_2D_global(grid_x, grid_y, w0, dt, num_steps, mu, basis, basis2,
                                       W_global, q_p_train, q_s_train, epsilon, scaler, kernel_type="gaussian"):
    """
    Solves the 2D inviscid Burgers' equations using a Reduced-Order Model (ROM)
    augmented with Proper Orthogonal Decomposition (POD) and global Radial Basis Functions (RBF).
    
    Parameters:
    - grid_x, grid_y: Arrays defining the grid points in the x and y directions.
    - w0: Initial state vector (flattened for both u and v components).
    - dt: Time step size.
    - num_steps: Number of time steps to simulate.
    - mu: Parameter vector [mu1, mu2].
    - basis: Primary POD modes matrix (\(\mathbf{V}\)).
    - basis2: Secondary POD modes matrix (\(\mathbf{\bar{V}}\)).
    - W_global: Precomputed global RBF weights matrix.
    - q_p_train, q_s_train: Training data for RBF interpolation.
    - epsilon: Shape parameter for RBF.
    - scaler: Scaler for normalization (e.g., MinMaxScaler or StandardScaler).
    - kernel_type: Type of RBF kernel to use (e.g., "gaussian").
    
    Returns:
    - snaps: Array of solution snapshots at each time step.
    - (num_its, jac_time, res_time, ls_time): Performance metrics.
    """

    # -----------------------------------
    # 1. Operators Setup
    # -----------------------------------
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initial conditions
    y0 = basis.T @ w0  # Project w0 onto the POD basis
    w0_reconstructed = decode_rbf_global(y0, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)

    nred = y0.shape[0]
    snaps = np.zeros((w0_reconstructed.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0_reconstructed
    red_coords[:, 0] = y0

    wp = w0_reconstructed.copy()
    yp = y0.copy()

    # Decode function using global RBF
    def decode_func(x):
        return decode_rbf_global(x, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)

    # Jacobian function for global RBF
    def jac_rbf_func(x):
        return jac_rbf_global(x, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type, echo_level=0)

    print(f"Running POD-RBF Global of size {nred} for mu1={mu[0]}, mu2={mu[1]}")

    # Time-Stepping Loop
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec)

        def jac(w):
            return inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye)

        print(f" ... Working on timestep {i}")
        t0 = time.time()

        # Solve using Gauss-Newton for POD-RBF
        y, resnorms, times = gauss_newton_pod_rbf(
            res, jac, yp, decode_func, jac_rbf_func
        )
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        # Reconstruct the full state
        w_reconstructed = decode_func(y)

        red_coords[:, i + 1] = y
        snaps[:, i + 1] = w_reconstructed
        wp = w_reconstructed
        yp = y

    return snaps, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_pod_rbf_2D_global_ecsw(grid_x, grid_y, w0, dt, num_steps, mu, basis, basis2,
                                            W_global, q_p_train, q_s_train, weights, epsilon, scaler, kernel_type='gaussian'):
    """
    Use a first-order Godunov spatial discretization and a second-order trapezoid rule
    time integrator to solve an LSPG manifold PROM for a parameterized inviscid 2D Burgers'
    problem with a source term, using global RBF interpolation with ECSW weights.

    Parameters:
    - grid_x, grid_y: Spatial grids.
    - w0: Initial condition.
    - dt: Time step size.
    - num_steps: Number of time steps.
    - mu: List of parameters [mu1, mu2].
    - basis: POD basis (U_p).
    - basis2: Secondary basis (U_s).
    - W_global: Precomputed global RBF weights matrix.
    - q_p_train: Training data for principal modes.
    - q_s_train: Training data for secondary modes.
    - weights: ECSW weights for sampled nodes.
    - epsilon: RBF parameter.
    - scaler: MinMaxScaler for normalization.
    - kernel_type: RBF kernel type ('gaussian', 'imq', 'linear', 'multiquadric').

    Returns:
    - red_coords: Reduced coordinates over time.
    - stats: Tuple (num_iterations, jac_time, res_time, ls_time).
    """
    
    # stuff for operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    # Mesh sampling based on ECSW weights
    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size / 2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()

    sample_weights = np.concatenate((weights, weights))[sample_inds]

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initial conditions
    y0 = basis.T @ w0  # Project w0 onto the POD basis
    w0_reconstructed = decode_rbf_global(y0, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)

    nred = y0.shape[0]
    snaps = np.zeros((w0_reconstructed.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0_reconstructed
    red_coords[:, 0] = y0

    wp = w0_reconstructed.copy()
    yp = y0.copy()

    # Reduced basis for sampled nodes
    idx = np.concatenate((augmented_sample, int(w0_reconstructed.shape[0] / 2) + augmented_sample))
    wp = wp[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]

    # Decode function using global RBF
    def decode_func(x):
        return decode_rbf_global(x, W_global, q_p_train, V, Vbar, epsilon, scaler, kernel_type, echo_level=0)

    # Jacobian function for global RBF
    def jac_rbf_func(x):
        return jac_rbf_global(x, W_global, q_p_train, q_s_train, V, Vbar, epsilon, scaler, kernel_type, echo_level=0)

    print(f"Running POD-RBF M-ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")
    lbc = None
    src = None
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)

    # Initialize boundary conditions and source terms
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    wall_clock_time = 0.0

    # Time-stepping loop
    for i in range(num_steps):
        def res(w):
            return inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc, src)

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_sample)

        print(f" ... Working on timestep {i}")
        t0 = time.time()

        # Solve using Gauss-Newton for POD-RBF
        y, resnorms, times = gauss_newton_pod_rbf_ecsw(
            res, jac, yp, decode_func, jac_rbf_func, sample_inds, augmented_sample, sample_weights
        )
        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        # Reconstruct the full state
        w_reconstructed = decode_rbf_global(y, W_global, q_p_train, V, Vbar, epsilon, scaler, kernel_type, echo_level=0)

        red_coords[:, i + 1] = y
        wp = w_reconstructed
        yp = y
        wall_clock_time += time.time() - t0

    return red_coords, (num_its, jac_time, res_time, ls_time)

def inviscid_burgers_pod_gp_2D_ecsw(
    grid_x, grid_y, w0, dt, num_steps, mu,
    basis, basis2,
    gp_model,           # CHANGED: replaces W_global
    weights,
    scaler,             # CHANGED: keep for q_p normalization
    q_snaps
):
    """
    Adaptation of the RBF-based M-ROM to a GP-based M-ROM for 2D inviscid Burgers with ECSW.

    Parameters:
    - grid_x, grid_y: Spatial grids.
    - w0: Initial condition.
    - dt: Time step size.
    - num_steps: Number of time steps.
    - mu: List of parameters [mu1, mu2].
    - basis: Primary POD basis (analogous to U_p).
    - basis2: Secondary POD basis (analogous to U_s).
    - gp_model: Trained Gaussian Process model (multi-output).
    - weights: ECSW weights for sampled nodes.
    - scaler: MinMaxScaler for q_p (if used for normalizing the primary coordinates).

    Returns:
    - red_coords: Reduced coordinates over time.
    - stats: Tuple (num_iterations, jac_time, res_time, ls_time).
    """

    # -------------------------------------------------------------------------
    # The ECSW + operator assembly logic remains identical to the RBF version.
    # -------------------------------------------------------------------------
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    Eye = Eye.tolil()
    JDxec = JDxec.tolil()
    JDyec = JDyec.tolil()

    sample_inds = np.where(weights != 0)[0]
    augmented_sample = generate_augmented_mesh(grid_x, grid_y, sample_inds)

    Eye = sp.identity(int(w0.size / 2)).tocsr()
    Eye = Eye[sample_inds, :][:, augmented_sample]
    Eye = sp.bmat([[Eye, None], [None, Eye]]).tocsr()

    JDxec_ecsw = JDxec[sample_inds, :][:, augmented_sample]
    JDyec_ecsw = JDyec[sample_inds, :][:, augmented_sample]
    JDyec = JDyec_ecsw.tocsr()
    JDxec = JDxec_ecsw.tocsr()

    sample_weights = np.concatenate((weights, weights))[sample_inds]

    num_its = 0
    jac_time = 0
    res_time = 0
    ls_time = 0

    # -------------------------------------------------------------------------
    # Initial condition: project onto POD basis, then decode with GP
    # -------------------------------------------------------------------------
    y0 = basis.T @ w0  # Project w0 onto the primary POD basis
    # CHANGED: replaced decode_rbf_global(...) with decode_gp(...)
    w0_reconstructed = decode_gp(y0, gp_model, basis, basis2, scaler)

    nred = y0.shape[0]
    snaps = np.zeros((w0_reconstructed.shape[0], num_steps + 1))
    red_coords = np.zeros((nred, num_steps + 1))
    snaps[:, 0] = w0_reconstructed
    red_coords[:, 0] = y0

    wp = w0_reconstructed.copy()
    yp = y0.copy()

    idx = np.concatenate((
        augmented_sample,
        int(w0_reconstructed.shape[0] / 2) + augmented_sample
    ))
    wp = wp[idx]

    V = basis[idx, :]
    Vbar = basis2[idx, :]

    # -------------------------------------------------------------------------
    # GP-based decode/jac functions
    # -------------------------------------------------------------------------
    # CHANGED: 'decode_func' references decode_gp instead of decode_rbf_global
    def decode_func(x):
        return decode_gp(x, gp_model, V, Vbar, scaler, echo_level=0)

    # CHANGED: new GP-based Jacobian function; in RBF approach we had jac_rbf_global
    def jac_gp_func(x):
        return jac_gp(x, gp_model, V, Vbar, scaler, echo_level=0)

    print(f"Running POD-GP M-ROM of size {nred} for mu1={mu[0]}, mu2={mu[1]}")
    lbc = None
    src = None
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)

    # -------------------------------------------------------------------------
    # Boundary conditions, source initialization
    # -------------------------------------------------------------------------
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]

    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    wall_clock_time = 0.0

    # -------------------------------------------------------------------------
    # Time-stepping loop with ECSW
    # -------------------------------------------------------------------------
    for i in range(num_steps):

        def res(w):
            return inviscid_burgers_res2D_ecsw(
                w, grid_x, grid_y, dt, wp, mu,
                JDxec, JDyec, sample_inds, augmented_sample,
                lbc, src
            )

        def jac(w):
            return inviscid_burgers_exact_jac2D_ecsw(
                w, dt, JDxec, JDyec, Eye,
                sample_inds, augmented_sample
            )

        print(f" ... Working on timestep {i}")
        t0 = time.time()

        # CHANGED: now calls a GP-based Gauss-Newton solver
        # Original was gauss_newton_pod_rbf_ecsw
        y, resnorms, times = gauss_newton_pod_gp_ecsw(
            res, jac, yp, decode_func, jac_gp_func,
            sample_inds, augmented_sample, sample_weights
        )

        jac_timep, res_timep, ls_timep = times
        num_its += len(resnorms)
        jac_time += jac_timep
        res_time += res_timep
        ls_time += ls_timep

        if i % 1000594 == 0 or i < 0:
            print(f"This step was given: {i}")
            y = q_snaps[:,i+1]

        # CHANGED: decode the new solution with GP
        w_reconstructed = decode_gp(y, gp_model, V, Vbar, scaler)

        red_coords[:, i + 1] = y
        wp = w_reconstructed
        yp = y
        wall_clock_time += time.time() - t0

    return red_coords, (num_its, jac_time, res_time, ls_time)

def decode_rbf_nearest_neighbors(x, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type='gaussian'):
    """
    Reconstruct the full state vector using POD and RBF interpolation (nearest neighbors).

    Parameters:
    - x: Reduced coordinates.
    - epsilon: Current epsilon for RBF.
    - neighbors: Number of nearest neighbors for RBF.
    - kdtree: KDTree to find nearest neighbors.
    - q_p_train: Training data for principal modes (q_p).
    - q_s_train: Training data for secondary modes (q_s).
    - basis: U_p matrix from POD.
    - basis2: U_s matrix from POD.
    - scaler: Scaler for normalization (e.g., StandardScaler or MinMaxScaler).
    - kernel_type: Type of RBF kernel to use ('gaussian', 'imq', 'linear', 'multiquadric').

    Returns:
    - Reconstructed full state vector.
    """

    if kernel_type == 'gaussian':
        q_s_pred = RBFUtils.interpolate_with_rbf_nearest_neighbors_dynamic_gaussian(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'imq':
        q_s_pred = RBFUtils.interpolate_with_rbf_nearest_neighbors_dynamic_imq(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'linear':
        q_s_pred = RBFUtils.interpolate_with_rbf_nearest_neighbors_dynamic_linear(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'multiquadric':
        q_s_pred = RBFUtils.interpolate_with_rbf_nearest_neighbors_dynamic_multiquadric(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    return basis @ x + basis2 @ q_s_pred

def jac_rbf_nearest_neighbors(x, kdtree, q_p_train, q_s_train, basis, basis2, epsilon, neighbors, scaler, kernel_type='gaussian'):
    """
    Compute the full Jacobian V = U_p + U_s * J_RBF (nearest neighbors).

    Parameters:
    - x: Reduced coordinates (q_p).
    - kdtree: KDTree to find nearest neighbors.
    - q_p_train: Training data for principal modes (q_p).
    - q_s_train: Training data for secondary modes (q_s).
    - basis: U_p matrix from POD.
    - basis2: U_s matrix from POD.
    - epsilon: Current epsilon for RBF.
    - neighbors: Number of nearest neighbors for RBF.
    - scaler: Scaler for normalization (e.g., StandardScaler or MinMaxScaler).
    - kernel_type: Type of RBF kernel to use ('gaussian', 'imq', 'linear', 'multiquadric').

    Returns:
    - Full Jacobian V with respect to reduced coordinates.
    """
    if kernel_type == 'gaussian':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_nearest_neighbors_dynamic_gaussian(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'imq':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_nearest_neighbors_dynamic_imq(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'linear':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_nearest_neighbors_dynamic_linear(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    elif kernel_type == 'multiquadric':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_nearest_neighbors_dynamic_multiquadric(
            kdtree, q_p_train, q_s_train, x, epsilon, neighbors, scaler, echo_level=0)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    return basis + basis2 @ rbf_jacobian

def decode_rbf_global(x, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type='gaussian', echo_level=0):
    """
    Reconstruct the full state vector using POD and global RBF interpolation.

    Parameters:
    - x: Reduced coordinates.
    - W_global: Precomputed global RBF weights.
    - q_p_train: Training data for principal modes.
    - basis, basis2: POD matrices for reconstruction.
    - epsilon: RBF shape parameter.
    - scaler: Normalization scaler.
    - kernel_type: RBF type ('gaussian', 'imq', 'linear', 'multiquadric').
    - echo_level: Verbosity level.

    Returns:
    - Full state vector.
    """
    # Perform global RBF interpolation
    if kernel_type == 'gaussian':
        q_s_pred = RBFUtils.interpolate_with_rbf_global_gaussian(x, q_p_train, W_global, epsilon, scaler, echo_level)
    elif kernel_type == 'imq':
        q_s_pred = RBFUtils.interpolate_with_rbf_global_imq(x, q_p_train, W_global, epsilon, scaler, echo_level)
    elif kernel_type == 'linear':
        q_s_pred = RBFUtils.interpolate_with_rbf_global_linear(x, q_p_train, W_global, scaler, echo_level)
    elif kernel_type == 'multiquadric':
        q_s_pred = RBFUtils.interpolate_with_rbf_global_multiquadric(x, q_p_train, W_global, epsilon, scaler, echo_level)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # Reconstruct the full state vector
    return basis @ x + basis2 @ q_s_pred

def jac_rbf_global(x, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type='gaussian', echo_level = 0):
    """
    Compute the full Jacobian V = U_p + U_s * J_RBF (global).

    Parameters:
    - x: Reduced coordinates.
    - W_global: Precomputed global RBF weights matrix.
    - q_p_train: Training data for principal modes (q_p).
    - q_s_train: Training data for secondary modes (q_s).
    - basis: U_p matrix from POD.
    - basis2: U_s matrix from POD.
    - epsilon: Shape parameter for RBF.
    - scaler: Scaler for normalization (e.g., StandardScaler or MinMaxScaler).
    - kernel_type: Type of RBF kernel to use ('gaussian', 'imq', 'linear', 'multiquadric').

    Returns:
    - Full Jacobian V with respect to reduced coordinates.
    """

    # Normalize the input reduced coordinates
    x_normalized = scaler.transform(x.reshape(1, -1))

    # Compute RBF Jacobian globally
    if kernel_type == 'gaussian':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_global_gaussian(x_normalized, q_p_train, W_global, epsilon, scaler, echo_level=echo_level)
    elif kernel_type == 'imq':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_global_imq(x_normalized, q_p_train, W_global, epsilon, scaler, echo_level=echo_level)
    elif kernel_type == 'linear':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_global_linear(x_normalized, q_p_train, W_global, epsilon, scaler, echo_level=echo_level)
    elif kernel_type == 'multiquadric':
        rbf_jacobian = RBFUtils.compute_rbf_jacobian_global_multiquadric(x_normalized, q_p_train, W_global, epsilon, scaler, echo_level=echo_level)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # Compute the full Jacobian
    return basis + basis2 @ rbf_jacobian

def decode_gp(x, gp_model, basis, basis2, scaler, use_custom_predict=True, echo_level=0):
    """
    Reconstruct the full state vector using POD and a trained Gaussian Process (GP).

    Parameters:
    ----------
    x : ndarray
        Primary-mode reduced coordinates of shape (r_p,) -- i.e., the "primary" part.
    gp_model : GaussianProcessRegressor (multi-output)
        Trained GP model mapping primary coords -> secondary coords.
    basis : ndarray
        Primary POD basis (shape: [n_dofs, r_p]).
    basis2 : ndarray
        Secondary POD basis (shape: [n_dofs, r_s]).
    scaler : MinMaxScaler
        Scaler used for normalizing primary coordinates.
    echo_level : int, optional
        Verbosity level (unused here, but kept for consistency).

    Returns:
    --------
    full_state : ndarray
        Reconstructed full state vector of shape (n_dofs,).
    """

    x_in = x.reshape(1, -1)
    x_scaled = scaler.transform(x_in)

    t0 = time.time()

    if not use_custom_predict:
        # Standard scikit-learn approach
        q_s_pred = gp_model.predict(x_scaled).ravel()
    else:
        # Custom approach
        X_train_ = gp_model.X_train_
        alpha_   = gp_model.alpha_
        kernel_  = gp_model.kernel_

        k_vec = kernel_(X_train_, x_scaled).ravel()  # shape: (N,)
        q_s_pred = k_vec @ alpha_                    # shape: (r_s,)

    if echo_level > 0:
        print(f"Time to interpolate q_s: {time.time() - t0:.6f} seconds")

    # Now combine primary coords x with predicted secondary coords
    full_state = basis @ x + basis2 @ q_s_pred

    return full_state

def jac_gp_central_difference(
    x, gp_model, basis, basis2, scaler,
    fd_eps=1e-6, echo_level=0,
    use_custom_predict=True
):
    """
    Compute the full Jacobian V = U_p + U_s * d(q_s)/d(x)
    for a POD-GP method, where q_s = GP(q_p).

    This function uses a central-difference scheme to approximate the partial
    derivatives of the GP model output (secondary coordinates) w.r.t. the
    original (unscaled) primary coordinates x. It applies the chain rule
    for an internal scaling transform x_scaled.

    Parameters
    ----------
    x : ndarray of shape (r_p,)
        The current reduced primary POD coordinates (unscaled).
    gp_model : GaussianProcessRegressor (multi-output)
        Trained GP model mapping primary coords -> secondary coords.
        This model expects scaled inputs unless we do a custom approach.
    basis : ndarray of shape (n_dofs, r_p)
        Primary POD basis, U_p.
    basis2 : ndarray of shape (n_dofs, r_s)
        Secondary POD basis, U_s.
    scaler : MinMaxScaler (or similar)
        Scaler used to normalize the primary coordinates; must match training.
    fd_eps : float, optional
        Step size for central differencing in scaled space.
    echo_level : int, optional
        Verbosity level (0=quiet).
    use_custom_predict : bool, optional
        If True, we manually compute k_vec @ alpha_ instead of calling gp_model.predict(...).
        Default = False (call scikit-learn's predict).

    Returns
    -------
    full_jac : ndarray of shape (n_dofs, r_p)
        The Jacobian of the decoded full state w.r.t. the original (unscaled)
        reduced primary coords: V = basis + basis2 @ (d q_s / d x).
    """

    import time
    t0 = time.time()

    # --- Step 1: Reshape + scale x -------------------------------------------
    x_in = x.reshape(1, -1)              # (1, r_p)
    x_scaled = scaler.transform(x_in)    # shape (1, r_p)
    scale_factors = scaler.scale_        # shape (r_p,)

    # We'll parse internal GP data if custom predict is used
    if use_custom_predict:
        X_train_ = gp_model.X_train_
        alpha_   = gp_model.alpha_
        kernel_  = gp_model.kernel_

    # --- Step 2: Baseline predict (not strictly needed, but 
    #     we compute it for reference if we want or for comparison)
    #     In central differencing, we typically don't strictly need a base
    #     but it can be useful if we want to measure cost or do checks.
    #     We'll store it just as info (not used for the difference).
    baseline_start = time.time()
    if not use_custom_predict:
        q_s_base = gp_model.predict(x_scaled).ravel()
    else:
        k_vec = kernel_(X_train_, x_scaled).ravel()
        q_s_base = k_vec @ alpha_
    baseline_time = time.time() - baseline_start

    # --- Step 3: Central differencing loop -----------------------------------
    r_p = x_scaled.shape[1]
    # We'll do x_plus and x_minus per dimension j
    # Then compute (q_s_plus - q_s_minus)/(2*fd_eps) * scale_factor
    dq_s_dxp = None

    loop_start = time.time()
    dq_s_dxp = np.zeros((q_s_base.size, r_p))

    half_eps = 0.5 * fd_eps
    for j in range(r_p):
        # build x_plus, x_minus
        x_plus  = x_scaled.copy()
        x_minus = x_scaled.copy()

        x_plus[0, j]  += half_eps
        x_minus[0, j] -= half_eps

        # q_s_plus
        if not use_custom_predict:
            q_s_plus  = gp_model.predict(x_plus).ravel()
            q_s_minus = gp_model.predict(x_minus).ravel()
        else:
            # custom approach for x_plus
            k_vec_plus  = kernel_(X_train_, x_plus).ravel()
            q_s_plus    = k_vec_plus @ alpha_
            # custom approach for x_minus
            k_vec_minus = kernel_(X_train_, x_minus).ravel()
            q_s_minus   = k_vec_minus @ alpha_

        # difference in scaled space
        dq_s_dxp_scaled = (q_s_plus - q_s_minus) / (2.0 * half_eps)

        # chain rule => multiply by scale_factors[j] to get derivative wrt unscaled x_j
        dq_s_dxp[:, j] = dq_s_dxp_scaled * scale_factors[j]

    loop_time = time.time() - loop_start

    # --- Step 4: Combine with POD bases --------------------------------------
    combine_start = time.time()
    # full_jac = U_p + U_s @ dq_s_dxp
    # shape => U_p: (n_dofs, r_p)
    #           U_s: (n_dofs, r_s)
    #    dq_s_dxp: (r_s, r_p)
    full_jac = basis + basis2 @ dq_s_dxp
    combine_time = time.time() - combine_start

    total_time = time.time() - t0
    if echo_level > 0:
        print("[jac_gp_central_difference] timing breakdown:")
        print(f"  Step 1 (scale input): {0:.6f} s (not individually timed)")
        print(f"  Step 2 (baseline predict): {baseline_time:.6f} s")
        print(f"  Step 3 (central diff loop): {loop_time:.6f} s")
        print(f"  Step 4 (combine w/ bases): {combine_time:.6f} s")
        print(f"  Total time: {total_time:.6f} s")

    return full_jac

def jac_gp(
    x, gp_model, basis, basis2, scaler,
    fd_eps=1e-6, echo_level=0,
    use_custom_predict=True
):
    """
    Forward-difference approximation in a loop (not batched).
    Slower than batch but uses half the calls of central difference.

    Now instrumented with additional timing prints to diagnose performance.
    The 'use_custom_predict' parameter lets you switch between scikit-learn's 
    gp_model.predict(...) or a custom approach (k_vec @ alpha_) for each step.
    """

    import time

    # Measure overall start time
    total_start_time = time.time()

    # --- Step 1: Reshape and scale x -----------------------------------------
    step1_start = time.time()
    x_in = x.reshape(1, -1)
    x_scaled = scaler.transform(x_in)
    scale_factors = scaler.scale_
    step1_time = time.time() - step1_start

    # --- Step 2: Baseline prediction -----------------------------------------
    step2_start = time.time()
    if not use_custom_predict:
        # Standard scikit-learn approach
        q_s_base = gp_model.predict(x_scaled).ravel()
    else:
        # Custom approach: read gp internals once outside the loop
        X_train_ = gp_model.X_train_
        alpha_   = gp_model.alpha_
        kernel_  = gp_model.kernel_
        k_vec = kernel_(X_train_, x_scaled).ravel()
        q_s_base = k_vec @ alpha_  # shape (r_s,)
    step2_time = time.time() - step2_start

    # --- Step 3: Forward difference loop -------------------------------------
    step3_start = time.time()
    r_p = x_in.shape[1]
    r_s = q_s_base.size
    dq_s_dxp = np.zeros((r_s, r_p))

    iteration_times = []

    # If using the custom approach, let's fetch gp internals once
    if use_custom_predict:
        X_train_ = gp_model.X_train_
        alpha_   = gp_model.alpha_
        kernel_  = gp_model.kernel_

    for j in range(r_p):
        iter_start = time.time()
        x_plus = x_scaled.copy()
        x_plus[0, j] += fd_eps

        if not use_custom_predict:
            # Standard scikit-learn predict
            q_s_plus = gp_model.predict(x_plus).ravel()
        else:
            # Custom kernel approach
            k_vec_plus = kernel_(X_train_, x_plus).ravel()
            q_s_plus = k_vec_plus @ alpha_

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)

        dq_s_scaled = (q_s_plus - q_s_base) / fd_eps
        dq_s_dxp[:, j] = dq_s_scaled * scale_factors[j]

    step3_time = time.time() - step3_start

    # --- Step 4: Combine with POD bases --------------------------------------
    step4_start = time.time()
    # full_jac = U_p + U_s @ (dq_s_dxp)
    # U_p = basis, U_s = basis2
    full_jac = basis + basis2 @ dq_s_dxp
    step4_time = time.time() - step4_start

    # --- Print times if echo_level > 0 ---------------------------------------
    total_time = time.time() - total_start_time
    if echo_level > 0:
        print("[jac_gp] Timing breakdown:")
        print(f"  Step 1 (reshape/scale): {step1_time:.6f} s")
        print(f"  Step 2 (baseline predict): {step2_time:.6f} s")
        print(f"  Step 3 (loop + forward diffs): {step3_time:.6f} s")
        for i, t_iter in enumerate(iteration_times):
            print(f"    - iteration {i}: {t_iter:.6f} s (predict call)")
        print(f"  Step 4 (combine w/ bases): {step4_time:.6f} s")
        print(f"  Total Jacobian time: {total_time:.6f} s")

    return full_jac

def jac_gp_analytical_in_progress(
    x, gp_model, basis, basis2, scaler, echo_level=0
):
    """
    Compute the Jacobian d(w)/d(x) for a POD-GP model with Matern(nu=1.5) kernel, 
    using an analytical approach (no finite differences).

    Parameters
    ----------
    x : ndarray of shape (r_p,)
        The original (unscaled) primary POD coordinates.
    gp_model : GaussianProcessRegressor
        Fitted GP model with Matern(nu=1.5). We assume no sum of kernels.
    basis : ndarray of shape (n_dofs, r_p)
        Primary POD basis, U_p.
    basis2 : ndarray of shape (n_dofs, r_s)
        Secondary POD basis, U_s.
    scaler : MinMaxScaler (or similar)
        Used to transform x -> x_scaled. Must match training scaling.
    echo_level : int
        Verbosity level.

    Returns
    -------
    full_jac : ndarray of shape (n_dofs, r_p)
        The Jacobian of the decoded full state w.r.t. the original (unscaled) x.
        i.e. d(w)/dx = U_p + U_s @ d(q_s)/dx.
    """
    import time
    t0 = time.time()

    # --- 1) Reshape & scale x
    x_in = x.reshape(1, -1)                # (1, r_p)
    x_scaled = scaler.transform(x_in)      # (1, r_p)
    scale_factors = scaler.scale_          # shape (r_p,)

    # --- 2) Parse kernel hyperparams
    # We assume the kernel is Matern(...) or ConstantKernel * Matern(...)
    kernel_ = gp_model.kernel_

    # E.g. if it's "ConstantKernel(...) * Matern(length_scale=..., nu=1.5)"
    # you might need to do kernel_.k2 for Matern or parse kernel_.get_params().
    # For simplicity, let's suppose you do:
    print(kernel_.get_params())
    length_scale = kernel_.get_params()['length_scale']   # float
    nu           = kernel_.get_params()['nu']            # should be 1.5
    variance     = 1.0  # default amplitude
    # If you have a ConstantKernel, might do something like:
    # amplitude = kernel_.k1.constant_value
    # or parse the product.

    # For Matern(nu=1.5), the kernel can be:
    # k(r) = amplitude * (1 + sqrt(3)*r/ell)*exp(-sqrt(3)*r/ell)
    # We parse amplitude either from kernel_ or from alpha_ if needed.

    # --- 3) Build the kernel vector & gradient wrt scaled x
    #  # train data
    X_train_ = gp_model.X_train_  # shape (N, r_p)
    alpha_   = gp_model.alpha_    # shape (N, r_s)
    # Possibly skip adding WhiteKernel noise since it's handled in alpha_.

    # We'll define a helper to compute the derivative d(k(r))/d(x) for each row

    def matern15_k_and_grad(x_sc, X_train, length_scale):
        """
        Returns:
         k_vec  : shape (N,) the Matern(1.5) kernel values
         dk_dX  : shape (N, r_p), the derivative w.r.t. scaled x
        x_sc is shape (1, r_p)
        """
        # un-batch x_sc => (r_p,)
        x_sc = x_sc.flatten()
        N = X_train.shape[0]
        r_p = x_sc.size

        k_vec = np.zeros(N)
        dk_dX = np.zeros((N, r_p))

        # In scaled domain, we interpret distance as Euclidean in scaled space
        # But if the kernel was fitted on scaled coords, length_scale is
        # the param. We'll do: r_i = ||(x_sc - X_train[i])|| in scaled space

        for i in range(N):
            diff = x_sc - X_train[i]   # shape (r_p,)
            r = np.linalg.norm(diff)   # float

            if r < 1e-14:
                # If x == X_train[i], derivative is 0 or we handle limit
                k_vec[i] = 1.0  # for r=0 => (1 + 0)*exp(0) = 1 * amplitude
                # derivative is 0 => no direction change if r=0
                dk_dX[i,:] = 0.0
            else:
                # Matern(1.5) => k(r) = (1 + sqrt(3)*r/ell)*exp(-sqrt(3)*r/ell)
                sqrt3 = np.sqrt(3.0)
                z = r / length_scale
                # k(r):
                kr = (1.0 + sqrt3*z) * np.exp(-sqrt3*z)  # ignoring amplitude for now

                # derivative wrt r:
                # d/d(r)[ (1 + sqrt3*z)*exp(-sqrt3*z ) ] * d(z)/d(r)
                # = ...
                # simpler version:
                #  d/d(r)  => derivative = ...
                # let's do direct approach:
                #   let A(r) = 1 + sqrt3*r/ell
                #   let B(r) = exp(-sqrt3*r/ell)
                #   => k(r) = A*B
                # d(k)/dr = A'(r)*B(r) + A(r)*B'(r)
                #   A'(r) = sqrt3/ell
                #   B'(r) = -sqrt3/ell * exp(-sqrt3*r/ell)
                # => derivative wrt r:
                #   dK/dr = sqrt3/ell * B  + A * [ -sqrt3/ell B ]
                #         = (sqrt3/ell)*B - (sqrt3/ell)*A*B
                #         = (sqrt3/ell)*B[1 - A]
                A = 1.0 + sqrt3*z
                B = np.exp(-sqrt3*z)
                dKdr = (sqrt3/length_scale)*B*(1.0 - A)

                k_vec[i] = kr  # ignoring amplitude => we can multiply amplitude later

                # chain rule wrt x_j: dK/dx_j = dK/dr * dr/dx_j
                # dr/dx_j = diff[j]/r
                grad_i = dKdr * (diff / r)
                dk_dX[i,:] = grad_i

        return k_vec, dk_dX

    # Compute kernel & grad in scaled space
    k_vec, dk_dX = matern15_k_and_grad(x_scaled, X_train_, length_scale)

    # If you have an amplitude => multiply k_vec, dk_dX by amplitude
    # e.g. amplitude = ...
    # k_vec *= amplitude
    # dk_dX *= amplitude

    # --- 4) Predicted q_s & its derivative wrt scaled x
    # q_s(x) = k_vec^T alpha_
    # => shape (r_s,)
    # derivative: d(q_s)/d(x_sc) = dk_dX^T alpha_
    # => shape (r_s, r_p)
    q_s = k_vec @ alpha_  # shape (r_s,)
    dq_s_d_xscaled = dk_dX.T @ alpha_  # (r_p, N) * (N, r_s) => (r_p, r_s)
    # but typically we want shape (r_s, r_p):
    dq_s_d_xscaled = dq_s_d_xscaled.T  # => (r_s, r_p)

    # --- 5) If your original x is unscaled, then d(q_s)/d x = d(q_s)/d x_scaled * d(x_scaled)/d x
    # chain rule:
    # x_scaled_j = scale_factors[j]*(x_j - min_j), ignoring offset for derivative
    # => d(x_scaled_j)/d x_j = scale_factors[j]
    # So we multiply each column j by scale_factors[j]
    for j in range(dq_s_d_xscaled.shape[1]):
        dq_s_d_xscaled[:, j] *= scale_factors[j]

    # --- 6) Combine with POD bases
    # w(x) = U_p x + U_s q_s(x)
    # => derivative wrt x => d(w)/dx = U_p + U_s * d(q_s)/dx
    # shape: U_p is (n_dofs, r_p), U_s is (n_dofs, r_s)
    # dq_s_d_x is (r_s, r_p)
    # => final is (n_dofs, r_p)
    # Then add them
    U_p_part = basis                     # shape (n_dofs, r_p)
    U_s_part = basis2 @ dq_s_d_xscaled   # shape (n_dofs, r_p)
    full_jac = U_p_part + U_s_part

    if echo_level > 0:
        print(f"[jac_gp_analytical_matern15] total time: {time.time() - t0:.6f} s")

    return full_jac

def newton_raphson(func, jac, x0, max_its=20, relnorm_cutoff=1e-12):
    """
    Newton-Raphson method for solving nonlinear systems of equations.

    Parameters:
    - func            : function - The residual function to be minimized (nonlinear system).
    - jac             : function - Function that computes the Jacobian matrix for the system.
    - x0              : ndarray  - Initial guess for the solution.
    - max_its         : int      - Maximum number of iterations (default is 20).
    - relnorm_cutoff  : float    - Relative norm threshold for convergence (default is 1e-12).

    Returns:
    - x               : ndarray  - The computed solution vector after convergence or max iterations.
    - resnorms        : list     - List of residual norms at each iteration.
    """

    # Initialize the solution with the initial guess x0
    x = x0.copy()

    # Compute the initial residual norm
    init_norm = np.linalg.norm(func(x0))

    # Initialize a list to store residual norms at each iteration
    resnorms = []

    # Start the Newton-Raphson iteration loop
    for i in range(max_its):
        # Compute the current residual norm based on the function func
        resnorm = np.linalg.norm(func(x))
        resnorms.append(resnorm)  # Store the residual norm

        # Check if the relative residual norm is below the cutoff threshold (convergence check)
        if resnorm / init_norm < relnorm_cutoff:
            print('{}: {:3.2e}'.format(i, resnorm / init_norm))  # Print iteration and relative residual norm
            break  # Stop the iteration if convergence is achieved

        # Compute the Jacobian matrix at the current solution x
        J = jac(x)

        # Evaluate the residual function at the current solution x
        f = func(x)

        # Solve the linear system J * delta_x = -f to update the solution (delta_x = J^(-1) * (-f))
        x -= sp.linalg.spsolve(J, f)  # Use sparse linear solver for efficiency

    # Return the final solution x and the list of residual norms for monitoring
    return x, resnorms

def gauss_newton_LSPG(func, jac, basis, y0, max_its=20, relnorm_cutoff=1e-5, min_delta=0.1):
    """
    Gauss-Newton solver for Least-Squares Petrov-Galerkin (LSPG) projection.

    Parameters:
    - func           : Function to compute the residual.
    - jac            : Function to compute the Jacobian.
    - basis          : Reduced basis matrix.
    - y0             : Initial guess for reduced coordinates.
    - max_its        : Maximum iterations (default 20).
    - relnorm_cutoff : Stop if relative residual norm is below this (default 1e-5).
    - min_delta      : Stop if change in residual norm is too small (default 0.1).

    Returns:
    - y              : Optimized reduced coordinates.
    - resnorms       : List of residual norms at each iteration.
    - times          : Tuple of times spent on Jacobian, residual, and least-squares solve.
    """
    
    # Timing variables to track the time spent on Jacobian, residual computation, and least-squares solve
    jac_time = 0
    res_time = 0
    ls_time = 0
    
    # Initialize reduced coordinates and corresponding full-order state
    y = y0.copy()         # Copy the initial guess for reduced coordinates
    w = basis.dot(y0)     # Full-order state approximation: w = Phi * y0
    
    # Compute the initial residual norm for the full-order state
    init_norm = np.linalg.norm(func(w))
    resnorms = []  # To store the residual norms at each iteration
    
    # Gauss-Newton iteration loop
    for i in range(max_its):
        # Compute the current residual norm and add it to the list
        resnorm = np.linalg.norm(func(w))
        resnorms += [resnorm]
        
        # Stopping criterion based on relative residual norm
        if resnorm / init_norm < relnorm_cutoff:
            break
        
        # Stopping criterion based on insufficient reduction of the residual norm
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        
        # Compute the Jacobian matrix (time-tracked)
        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0  # Accumulate Jacobian computation time
        
        # Compute the residual vector (time-tracked)
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0  # Accumulate residual computation time
        
        # Solve the least-squares problem using the reduced Jacobian (time-tracked)
        t0 = time.time()
        JV = J.dot(basis)  # Project the Jacobian into the reduced space
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)  # Solve for the update in reduced coordinates
        ls_time += time.time() - t0  # Accumulate least-squares solve time
        
        # Update the reduced coordinates and full-order state
        y += dy
        w = basis.dot(y)  # Update full-order state using the new reduced coordinates
    
    # Output the final iteration and relative residual norm
    print('iteration {}: relative norm {:3.2e}'.format(i, resnorm / init_norm))
    
    # Return the optimized reduced coordinates, residual norms, and the timing information
    return y, resnorms, (jac_time, res_time, ls_time)

import numpy as np
import torch

def gauss_newton_ae_LSPG(func, jac, decoder, z0, max_its=20, relnorm_cutoff=1e-5, min_delta=0.1):
    """
    Gauss-Newton solver for Least-Squares Petrov-Galerkin (LSPG) projection 
    for a nonlinear manifold autoencoder-based ROM.

    Parameters:
    - func           : Function to compute the full-order residual.
    - jac            : Function to compute the full-order Jacobian.
    - decoder        : Autoencoder decoder (PyTorch model).
    - z0             : Initial guess for reduced coordinates.
    - max_its        : Maximum iterations (default 20).
    - relnorm_cutoff : Stop if relative residual norm is below this (default 1e-5).
    - min_delta      : Stop if change in residual norm is too small (default 0.1).

    Returns:
    - z              : Optimized reduced coordinates.
    - resnorms       : List of residual norms at each iteration.
    - times          : Tuple of times spent on Jacobian, residual, and least-squares solve.
    """
    
    import time
    
    # Timing variables
    jac_time = 0
    res_time = 0
    ls_time = 0
    
    # Initialize reduced coordinates
    z = z0.copy()

    # Device configuration for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)

    # Compute initial residual norm
    w = decoder(torch.tensor(z, dtype=torch.float32, device=device)).detach().cpu().numpy()
    init_norm = np.linalg.norm(func(w))
    resnorms = []
    
    for i in range(max_its):
        # Compute residual
        t0 = time.time()
        R = func(w)
        res_time += time.time() - t0

        # Compute residual norm
        resnorm = np.linalg.norm(R)
        resnorms.append(resnorm)

        # Stopping criteria
        if resnorm / init_norm < relnorm_cutoff:
            break
        if len(resnorms) > 1 and abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta:
            break
        
        # Compute Jacobian of residual
        t0 = time.time()
        J_u = jac(w)  # Full model Jacobian
        jac_time += time.time() - t0

        # Compute decoder Jacobian J_g using automatic differentiation
        def decoder_function(z):
            z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=device)
            w_recon = decoder(z_tensor)
            return w_recon

        t0 = time.time()
        J_g_tensor = torch.autograd.functional.jacobian(decoder_function, torch.tensor(z, dtype=torch.float32, device=device))
        J_g = J_g_tensor.detach().cpu().numpy()
        ls_time += time.time() - t0

        # Compute the reduced Jacobian using pseudoinverse
        J_g_pinv = np.linalg.pinv(J_g)  # Compute J_g^+
        J_r = J_g_pinv @ J_u @ J_g  # Reduced Jacobian
        R_r = J_g_pinv @ R  # Reduced residual

        # Solve the least squares problem
        delta_z, _, _, _ = np.linalg.lstsq(J_r, -R_r, rcond=None)

        # Update reduced coordinates
        z += delta_z
        w = decoder(torch.tensor(z, dtype=torch.float32, device=device)).detach().cpu().numpy()

    print(f'Iteration {i}: Relative norm {resnorm / init_norm:.2e}')
    
    return z, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_ECSW_2D(func, jac, basis, y0, sample_inds, augmented_sample, weight,
                      stepsize=1, max_its=20, relnorm_cutoff=1e-5, min_delta=1E-1):
    y = y0.copy()
    w = basis.dot(y0)

    weights = np.concatenate((weight, weight))
    init_norm = np.linalg.norm(func(w) * weights)
    resnorms = []
    jac_time = 0
    res_time = 0
    ls_time = 0

    for i in range(max_its):
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms += [resnorm]
        if resnorm/init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        t0 = time.time()
        J = jac(w)
        jac_time += time.time() - t0
        JV = J @ basis
        dw = sp.spdiags(weights, 0, weights.size, weights.size)
        JVw = dw@JV
        t0 = time.time()
        f = func(w)
        res_time += time.time() - t0
        t0 = time.time()
        fw = f * weights
        dy = np.linalg.lstsq(JVw, -fw, rcond=None)[0]
        ls_time += time.time() - t0
        y = y + stepsize*dy

        w = basis.dot(y)

    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_rnm(func, jac, y0, decode, jacfwdfunc,
                     max_its=20, relnorm_cutoff=1e-5,
                     lookback=10,
                     min_delta=0.1):
    """
    Performs the Gauss-Newton iterative method to solve for reduced coordinates
    in the LSPG-PROM framework augmented with POD-ANN.

    Parameters:
    - func: Function to compute the residual F(w).
    - jac: Function to compute the Jacobian J(w).
    - y0: Initial guess for reduced coordinates (torch.Tensor).
    - decode: Function to reconstruct full state from reduced coordinates.
    - jacfwdfunc: Function to compute Jacobian of decode with respect to y.
    - max_its: Maximum number of iterations.
    - relnorm_cutoff: Relative residual norm for convergence.
    - lookback: (Unused) Placeholder for potential future use.
    - min_delta: Minimum relative improvement in residual norm to continue.

    Returns:
    - y: Updated reduced coordinates after convergence.
    - resnorms: List of residual norms at each iteration.
    - (jac_time, res_time, ls_time): Timing metrics for performance.
    """

    # Initialize timing counters
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Clone the initial reduced coordinates to avoid modifying y0
    y = y0.detach().clone()

    with torch.no_grad():
        # Reconstruct the full state from initial reduced coordinates
        w = decode(y)

    # Calculate the initial residual norm for convergence checking
    init_norm = np.linalg.norm(func(w.squeeze().numpy()))
    resnorms = []

    for i in range(max_its):
        # Compute current residual norm
        resnorm = np.linalg.norm(func(w.squeeze().numpy()))
        resnorms.append(resnorm)

        # Check if relative residual norm is below the cutoff
        if resnorm / init_norm < relnorm_cutoff:
            break

        # Check for minimal improvement to stop iterations
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        # Time Jacobian computation
        t0 = time.time()
        J = jac(w.squeeze().numpy())       # Compute Jacobian J(w)
        V = jacfwdfunc(y).detach()         # Compute derivative of decode w.r.t y
        jac_time += time.time() - t0

        # Time residual computation
        t0 = time.time()
        f = func(w.squeeze().numpy())      # Compute residual F(w)
        res_time += time.time() - t0

        # Time least-squares solve
        t0 = time.time()
        JV = J.dot(V)                        # Project Jacobian onto reduced space
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)  # Solve for delta_y
        ls_time += time.time() - t0

        with torch.no_grad():
            # Update reduced coordinates with the solution delta_y
            y += torch.tensor(dy, dtype=torch.float)
            # Reconstruct the new full state from updated reduced coordinates
            w = decode(y, False)

    # Print the number of iterations and final relative residual norm
    print('{} iterations: {:3.2e} relative norm'.format(i, resnorm / init_norm))
    
    # Return the final reduced coordinates, residual norms, and timing metrics
    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_rnm_joshua(func, jac, y0, decode, jacfwdfunc,
                     max_its=20, relnorm_cutoff=1e-5,
                     lookback=10,
                     min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0

    y = y0.detach().clone()
    with torch.no_grad():
        w = decode(y)
    init_norm = np.linalg.norm(func(w.squeeze().numpy()))
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w.squeeze().numpy()))
        resnorms += [resnorm]
        if resnorm / init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()
        J = jac(w.squeeze().numpy())
        V = jacfwdfunc(y).detach()

        jac_time += time.time() - t0
        t0 = time.time()
        f = func(w.squeeze().numpy())
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(V)
        dy, lst_res, rank, sval = np.linalg.lstsq(JV, -f, rcond=None)
        ls_time += time.time() - t0
        with torch.no_grad():
            y += torch.tensor(dy, dtype=torch.float)
            w = decode(y, False)
    print('{} iterations: {:3.2e} relative norm'.format(i, resnorm / init_norm))
    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_rnm_ecsw(func, jac, y0, decode, jacfwdfunc,
                          sample_inds, augmented_sample, weight,
                     max_its=20, relnorm_cutoff=1e-5,
                     min_delta=0.1):
    jac_time = 0
    res_time = 0
    ls_time = 0

    y = y0.detach().clone()
    with torch.no_grad():
        w = decode(y, False)
    weights = np.concatenate((weight, weight))
    init_norm = np.linalg.norm(func(w) * weights)
    resnorm = init_norm
    i = 0
    resnorms = []
    for i in range(max_its):
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms += [resnorm]
        if resnorm / init_norm < relnorm_cutoff:
            break
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break
        t0 = time.time()
        if torch.is_tensor(w):
            wn = w.squeeze().numpy()
        else:
            wn = w

        J = jac(wn)
        V = jacfwdfunc(y).detach()

        jac_time += time.time() - t0
        t0 = time.time()
        f = func(wn)
        fw = f * weights
        res_time += time.time() - t0
        t0 = time.time()
        JV = J.dot(V)
        dw = sp.spdiags(weights, 0, weights.size, weights.size)
        JVw = dw @ JV
        dy, lst_res, rank, sval = np.linalg.lstsq(JVw, -fw, rcond=None)
        ls_time += time.time() - t0
        with torch.no_grad():
            y += torch.tensor(dy, dtype=torch.float)
            w = decode(y, False).squeeze().numpy()
    print('{} iterations: {:3.2e} relative norm'.format(i, resnorm / init_norm))
    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_pod_rbf(func, jac, y0, decode_rbf, jac_rbf,
                         max_its=10, relnorm_cutoff=1e-5, min_delta=0.1):
    """
    Performs the Gauss-Newton iterative method to solve for reduced coordinates
    in the LSPG-PROM framework augmented with POD-RBF.

    Parameters:
    - func: Function to compute the residual F(w).
    - jac: Function to compute the Jacobian J(w).
    - y0: Initial guess for reduced coordinates.
    - decode_rbf: Function to reconstruct full state from reduced coordinates.
    - jac_rbf: Function to compute Jacobian of decode with respect to reduced coordinates.
    - max_its: Maximum number of iterations.
    - relnorm_cutoff: Relative residual norm for convergence.
    - min_delta: Minimum relative improvement in residual norm to continue.

    Returns:
    - y: Updated reduced coordinates after convergence.
    - resnorms: List of residual norms at each iteration.
    - (jac_time, res_time, ls_time): Timing metrics for performance.
    """

    # Initialize timing counters
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initialize reduced coordinates
    y = y0.copy()

    # Reconstruct the full state from initial reduced coordinates
    w = decode_rbf(y)

    # Calculate the initial residual norm
    init_norm = np.linalg.norm(func(w))
    resnorms = []

    for i in range(max_its):
        # Compute current residual norm
        resnorm = np.linalg.norm(func(w))
        resnorms.append(resnorm)

        # Check for convergence
        if resnorm / init_norm < relnorm_cutoff:
            break

        # Check for minimal improvement
        if len(resnorms) > 1 and abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta:
            break

        # Time Jacobian computation
        t0 = time.time()
        J = jac(w)  # Full-order model Jacobian
        V = jac_rbf(y)  # POD-RBF Jacobian
        jac_time += time.time() - t0

        # Time residual computation
        t0 = time.time()
        f = func(w)  # Residual
        res_time += time.time() - t0

        # Time least-squares solve
        t0 = time.time()
        JV = J.dot(V)  # Project full-order Jacobian into reduced space
        dy, _, _, _ = np.linalg.lstsq(JV, -f, rcond=None)  # Solve for delta_y
        ls_time += time.time() - t0

        # Update reduced coordinates
        y += dy

        # Reconstruct the full state with updated reduced coordinates
        w = decode_rbf(y)

    print(f'{i} iterations: {resnorm / init_norm:.2e} relative norm')

    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_pod_rbf_ecsw(func, jac, y0, decode_rbf, jac_rbf,
                              sample_inds, augmented_sample, weights,
                              max_its=10, relnorm_cutoff=1e-5,
                              min_delta=0.1):
    """
    Gauss-Newton solver for the POD-RBF approach using ECSW.
    
    Parameters:
    - func: Residual function for the HDM.
    - jac: Jacobian function for the HDM.
    - y0: Initial guess for the reduced coordinates.
    - decode_rbf: Function to decode reduced coordinates to full state using POD-RBF.
    - jac_rbf: Function to compute the Jacobian of the POD-RBF reconstruction.
    - sample_inds: Indices of sampled ECSW nodes.
    - augmented_sample: Augmented sample of nodes from ECSW.
    - weights: ECSW weights for sampled nodes.
    - max_its: Maximum number of Gauss-Newton iterations.
    - relnorm_cutoff: Relative residual norm cutoff for convergence.
    - min_delta: Minimum relative improvement in residual norm to continue.
    
    Returns:
    - y: Updated reduced coordinates after convergence.
    - resnorms: List of residual norms at each iteration.
    - (jac_time, res_time, ls_time): Timing metrics for Jacobian, residual, and least-squares solve.
    """
    
    # Initialize timing counters
    jac_time = 0
    res_time = 0
    ls_time = 0

    # Initialize reduced coordinates
    y = y0.copy()

    # Reconstruct the full state from initial reduced coordinates
    w = decode_rbf(y)

    # Weights vector for ECSW
    weights = np.concatenate((weights, weights))

    # Calculate the initial residual norm
    init_norm = np.linalg.norm(func(w) * weights)
    resnorm = init_norm
    i = 0
    resnorms = []

    for i in range(max_its):
        # Compute current residual norm
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms.append(resnorm)

        # Check for convergence
        if resnorm / init_norm < relnorm_cutoff:
            break

        # Check for minimal improvement
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        # Time Jacobian computation
        t0 = time.time()
        J = jac(w)
        V = jac_rbf(y)
        jac_time += time.time() - t0

        # Time residual computation
        t0 = time.time()
        f = func(w)
        fw = f * weights
        res_time += time.time() - t0

        # Time least-squares solve
        t0 = time.time()
        JV = J.dot(V)
        dw = sp.spdiags(weights, 0, weights.size, weights.size)  # Diagonal matrix with weights
        JVw = dw @ JV
        dy, lst_res, rank, sval = np.linalg.lstsq(JVw, -fw, rcond=None)
        ls_time += time.time() - t0

        # Update reduced coordinates
        y += dy

        # Reconstruct the full state with updated reduced coordinates
        w = decode_rbf(y)

    print(f'{i} iterations: {resnorm / init_norm:.2e} relative norm')
    
    return y, resnorms, (jac_time, res_time, ls_time)

def gauss_newton_pod_gp_ecsw(func, jac, y0, decode_gp, jac_gp,
                             sample_inds, augmented_sample, weights,
                             max_its=10, relnorm_cutoff=1e-5,
                             min_delta=0.1):
    """
    Gauss-Newton solver for the POD-GP approach using ECSW.
    
    Parameters
    ----------
    func : callable
        Residual function for the HDM. E.g., inviscid_burgers_res2D_ecsw(...).
    jac : callable
        Jacobian function for the HDM. E.g., inviscid_burgers_exact_jac2D_ecsw(...).
    y0 : ndarray
        Initial guess for the reduced (primary) coordinates.
    decode_gp : callable
        Function to decode reduced coordinates to the full state using POD + GP.
    jac_gp : callable
        Function to compute the Jacobian of the POD-GP reconstruction.
    sample_inds : ndarray
        Indices of sampled ECSW nodes.
    augmented_sample : ndarray
        Augmented sample of nodes from ECSW.
    weights : ndarray
        ECSW weights for sampled nodes.
    max_its : int, optional
        Maximum number of Gauss-Newton iterations. Default is 10.
    relnorm_cutoff : float, optional
        Relative residual norm cutoff for convergence. Default is 1e-5.
    min_delta : float, optional
        Minimum relative improvement in residual norm to continue. Default is 0.1.
    
    Returns
    -------
    y : ndarray
        Updated reduced coordinates after convergence.
    resnorms : list of float
        Residual norms at each iteration.
    (jac_time, res_time, ls_time) : tuple of floats
        Timing metrics for Jacobian, residual, and least-squares solve.
    """

    # Initialize timing counters
    jac_time = 0.0
    res_time = 0.0
    ls_time = 0.0

    # Initialize reduced coordinates
    y = y0.copy()

    # Reconstruct the full state from initial reduced coordinates
    w = decode_gp(y)

    # Weights vector for ECSW (duplicated for two components if needed)
    weights = np.concatenate((weights, weights))

    # Calculate the initial residual norm
    init_norm = np.linalg.norm(func(w) * weights)
    resnorm = init_norm
    resnorms = []

    for i in range(max_its):
        # Compute current residual norm
        resnorm = np.linalg.norm(func(w) * weights)
        resnorms.append(resnorm)

        # Check for convergence
        if resnorm / init_norm < relnorm_cutoff:
            break

        # Check for minimal improvement
        if (len(resnorms) > 1) and (abs((resnorms[-2] - resnorms[-1]) / resnorms[-2]) < min_delta):
            break

        # Time Jacobian computation (HDM + GP)
        t0 = time.time()
        J = jac(w)       # HDM part
        V = jac_gp(y)    # POD-GP part
        jac_time += time.time() - t0

        # Time residual computation
        t0 = time.time()
        f = func(w)
        fw = f * weights
        res_time += time.time() - t0

        # Time least-squares solve
        t0 = time.time()
        JV = J.dot(V)
        dw = sp.spdiags(weights, 0, weights.size, weights.size)  # diagonal matrix with ECSW weights
        JVw = dw @ JV
        dy, lst_res, rank, sval = np.linalg.lstsq(JVw, -fw, rcond=None)
        ls_time += time.time() - t0

        # Update reduced coordinates
        y += dy

        # Reconstruct the full state with updated reduced coordinates
        w = decode_gp(y)

    print(f'{i} iterations: {resnorm / init_norm:.2e} relative norm')
    
    return y, resnorms, (jac_time, res_time, ls_time)

def make_ddx(grid_x):
    """
    Constructs a matrix to calculate the variation between adjacent points in the x-direction.
    """
    dx = grid_x[1:] - grid_x[:-1]  # Grid spacing
    return sp.spdiags([-np.ones(grid_x.size - 1) / dx, np.ones(grid_x.size - 1) / dx], [-1, 0],
                      grid_x.size - 1, grid_x.size - 1, 'lil')

def make_mid(grid_x):
    """
    Constructs a matrix to compute the average of two adjacent points in the x-direction.
    """
    return sp.spdiags([np.ones(grid_x.size - 1) / 2, np.ones(grid_x.size - 1) / 2], [-1, 0],
                      grid_x.size - 1, grid_x.size - 1, 'lil')

def make_2D_grid(x_low, x_up, y_low, y_up, num_cells_x, num_cells_y):
    """
    Creates a 2D grid with specified boundaries and number of cells.
    """
    grid_x = np.linspace(x_low, x_up, num_cells_x + 1)  # x-direction grid points
    grid_y = np.linspace(y_low, y_up, num_cells_y + 1)  # y-direction grid points
    return grid_x, grid_y

def get_ops(grid_x, grid_y):
    Dxec = make_ddx(grid_x)
    Dyec = make_ddx(grid_y)
    JDxec = sp.kron(sp.eye(grid_y.size - 1, grid_y.size - 1), Dxec)
    JDyec = sp.kron(sp.eye(grid_x.size - 1, grid_x.size - 1), Dyec)
    JDyec = JDyec.tocsr()
    idx = np.arange((grid_y.size - 1) * (grid_x.size - 1)).reshape(
        (grid_y.size - 1, grid_x.size - 1)).T.ravel()
    JDyec = JDyec[idx, :]
    JDyec = JDyec[:, idx]
    Eye = sp.identity(2 * (grid_x.size - 1) * (grid_y.size - 1))
    return Dxec, Dyec, JDxec, JDyec, Eye

def generate_augmented_mesh(grid_x, grid_y, sample_inds):
    augmented_sample = set(sample_inds)
    shp = (grid_y.size - 1, grid_x.size - 1)
    for i in sample_inds:
        r, c = np.unravel_index(i, shp)
        if r - 1 >= 0:
            # print('adding ({}, {}): {}'.format(r - 1, c, np.ravel_multi_index((r - 1, c), shp)))
            augmented_sample.add(np.ravel_multi_index((r - 1, c), shp))
        # else:
        #     print('({}, {}): out of bounds!'.format(r - 1, c))
        if c - 1 >= 0:
            # print('adding ({}, {}): {}'.format(r, c - 1, np.ravel_multi_index((r, c - 1), shp)))
            augmented_sample.add(np.ravel_multi_index((r, c - 1), shp))
        # else:
        #     print('({}, {}): out of bounds!'.format(r, c - 1))
        if c - 1 >= 0:
            idx = np.ravel_multi_index((r, c - 1), shp)
            if idx not in augmented_sample:
                print('({}, {}): missing a point!'.format(r, c - 1))
    augmented_sample = np.sort(np.array(list(augmented_sample)))
    return augmented_sample

def inviscid_burgers_res2D(w, grid_x, grid_y, dt, wp, mu, Dxec, Dyec):
    """
    Compute the residual for the 2D inviscid Burgers' equation.

    Parameters:
    w      : ndarray, current state vector (u and v components).
    grid_x : ndarray, grid points in the x direction.
    grid_y : ndarray, grid points in the y direction.
    dt     : float, time step size.
    wp     : ndarray, previous state vector (u and v components).
    mu     : list, parameter vector [mu1, mu2].
    Dxec   : sparse matrix, derivative operator in x direction.
    Dyec   : sparse matrix, derivative operator in y direction.

    Returns:
    residual : ndarray, computed residual vector for the current time step.
    """

    # Calculate grid spacings and midpoints
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2

    # Split state vector for u and v components
    u_idx = dx.size * dy.size
    up, u = wp[:u_idx].reshape(dy.size, dx.size), w[:u_idx].reshape(dy.size, dx.size)
    vp, v = wp[u_idx:].reshape(dy.size, dx.size), w[u_idx:].reshape(dy.size, dx.size)

    # Compute flux terms for u and v
    Fux, Fpux = (0.5 * np.square(u)).T, (0.5 * np.square(up)).T
    Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)
    Fuv, Fpuv = 0.5 * u * v, 0.5 * up * vp

    # Source term
    src = dt * 0.02 * np.exp(mu[1] * xc[None, :])

    # Residual for u and v components
    ru = u - up + 0.5 * dt * (Dxec @ (Fux + Fpux)).T + 0.5 * dt * Dyec @ (Fuv + Fpuv) - src
    ru[:, 0] -= 0.5 * dt * mu[0] ** 2 / dx  # Boundary condition at inlet (x = 0)
    rv = v - vp + 0.5 * dt * Dyec @ (Fvy + Fpvy) + 0.5 * dt * (Dxec @ (Fuv.T + Fpuv.T)).T

    # Concatenate residuals for u and v
    return np.concatenate((ru.ravel(), rv.ravel()))

def inviscid_burgers_res2D_alt(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec):
    """
    Computes the residual for the 2D inviscid Burgers' equation using fluxes and boundary conditions.
    
    Parameters:
    - w       : ndarray - Current state vector (contains both u and v components).
    - grid_x  : ndarray - Grid points in the x direction.
    - grid_y  : ndarray - Grid points in the y direction.
    - dt      : float - Time step size.
    - wp      : ndarray - Previous state vector (contains both u and v components).
    - mu      : list - Parameters [mu1, mu2] (mu1 related to boundary, mu2 to source term).
    - JDxec   : sparse matrix - x-direction derivative operator (for flux calculation).
    - JDyec   : sparse matrix - y-direction derivative operator (for flux calculation).
    
    Returns:
    - residual : ndarray - The residual vector for both u and v components at the current timestep.
    """

    # Calculate grid spacing in x and y directions
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]

    # Calculate cell-centered x coordinates for source term calculation
    xc = (grid_x[1:] + grid_x[:-1]) / 2

    # Split the state vector into u and v components for the current timestep
    u, v = np.split(w, 2)

    # Split the previous timestep state vector into u and v components
    up, vp = np.split(wp, 2)

    # Compute fluxes for u and v (nonlinear advection terms)
    Fux, Fpux = (0.5 * np.square(u)), (0.5 * np.square(up))  # Flux in x-direction for u
    Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)      # Flux in y-direction for v
    Fuv, Fpuv = 0.5 * u * v, 0.5 * up * vp                   # Cross flux term (u * v)
    FuvT, FpuvT = Fuv, Fpuv                                   # Transposed cross flux (for v calculation)

    # Compute the source term based on the parameter mu[1] and the x-coordinates
    src = dt * 0.02 * np.exp(mu[1] * xc)

    # Left boundary condition (inlet boundary at x = 0)
    lbc = np.zeros_like(u).reshape((dy.size, dx.size))        # Initialize boundary condition array
    lbc[:, 0] = 0.5 * dt * mu[0] ** 2 / dx                    # Set boundary condition for u at the left boundary (x = 0)
    lbc = lbc.ravel()                                         # Flatten the array for further use

    # Apply source term to all grid points in the y direction
    src = np.tile(src, dy.size)

    # Compute the residual for the u component:
    ru = u - up + 0.5 * dt * JDxec @ (Fux + Fpux) + \
         0.5 * dt * JDyec @ (Fuv + Fpuv) - src           # Residual includes fluxes and source term for u
    ru -= lbc                                           # Subtract the boundary condition from the residual

    # Compute the residual for the v component:
    rv = v - vp + 0.5 * dt * JDyec @ (Fvy + Fpvy) + \
         0.5 * dt * (JDxec @ (FuvT + FpuvT))            # Residual includes fluxes and coupling between u and v

    # Return the combined residuals for both u and v components as a single concatenated vector
    return np.concatenate((ru, rv))

def inviscid_burgers_res2D_ecsw(w, grid_x, grid_y, dt, wp, mu, JDxec, JDyec, sample_inds, augmented_sample, lbc=None, src=None):
    """
    Assumes either the full state w or w[augmented_sample] as the input...
    """
    if torch.is_tensor(w):
        w = w.detach().squeeze().numpy()
    if torch.is_tensor(wp):
        wp = wp.detach().squeeze().numpy()
    dx = grid_x[1:] - grid_x[:-1]
    dy = grid_y[1:] - grid_y[:-1]
    xc = (grid_x[1:] + grid_x[:-1]) / 2
    shp = (dy.size, dx.size)
    if lbc is None:
        lbc = np.zeros_like(sample_inds, dtype=np.float64)
        t = np.unravel_index(sample_inds, shp)
        for i, (r, c) in enumerate(zip(t[0], t[1])):
            if c == 0:
                lbc[i] = 0.5 * dt * mu[0] ** 2 / dx[0]
    if src is None:
        src = dt * 0.02 * np.exp(mu[1] * xc)
        src = np.tile(src, dy.size)
        src = src[sample_inds]

    u, v = np.split(w, 2)
    up, vp = np.split(wp, 2)
    if u.size > augmented_sample.size:
        Fux, Fpux = 0.5 * np.square(u[augmented_sample]), 0.5 * np.square(up[augmented_sample])
        Fvy, Fpvy = 0.5 * np.square(v[augmented_sample]), 0.5 * np.square(vp[augmented_sample])
        Fuv = 0.5 * u[augmented_sample] * v[augmented_sample]
        Fpuv = 0.5 * up[augmented_sample] * vp[augmented_sample]

        u = u[sample_inds]
        v = v[sample_inds]
        up = up[sample_inds]
        vp = vp[sample_inds]
    else:
        Fux, Fpux = 0.5 * np.square(u), 0.5 * np.square(up)
        Fvy, Fpvy = 0.5 * np.square(v), 0.5 * np.square(vp)
        Fuv = 0.5 * u * v
        Fpuv = 0.5 * up * vp

        overlap = np.isin(augmented_sample, sample_inds)
        u = u[overlap]
        v = v[overlap]
        up = up[overlap]
        vp = vp[overlap]

    ru = u - up + 0.5 * dt * JDxec @ (Fux + Fpux) + \
         0.5 * dt * JDyec @ (Fuv + Fpuv) - src
    ru -= lbc
    rv = v - vp + 0.5 * dt * JDyec @ (Fvy + Fpvy) + \
         0.5 * dt * (JDxec @ (Fuv + Fpuv))

    return np.concatenate((ru, rv))

def inviscid_burgers_exact_jac2D(w, dt, JDxec, JDyec, Eye):
    """
    Compute the Jacobian matrix for the 2D inviscid Burgers' equation.

    Parameters:
    w      : ndarray, state vector (u and v components).
    dt     : float, time step size.
    JDxec  : sparse matrix, x-direction derivative operator.
    JDyec  : sparse matrix, y-direction derivative operator.
    Eye    : sparse matrix, identity matrix for stability.

    Returns:
    Jacobian : sparse matrix, the computed Jacobian.
    """

    # Split the state vector w into u and v components
    u, v = np.split(w, 2)

    # Diagonal matrices from u and v
    ud = 0.5 * dt * sp.diags(u)
    vd = 0.5 * dt * sp.diags(v)

    # Jacobian block components
    ul = JDxec @ ud + 0.5 * JDyec @ vd  # Upper left block
    ur = 0.5 * JDyec @ ud               # Upper right block
    ll = 0.5 * JDxec @ vd               # Lower left block
    lr = JDyec @ vd + 0.5 * JDxec @ ud   # Lower right block

    # Combine into full Jacobian matrix
    return sp.bmat([[ul, ur], [ll, lr]]) + Eye

def inviscid_burgers_exact_jac2D_ecsw(w, dt, JDxec, JDyec, Eye, sample_inds, augmented_inds):
    u, v = np.split(w, 2)
    if u.size > augmented_inds.size:
        ud, vd = 0.5 * dt * sp.diags(u[augmented_inds]), 0.5 * dt * sp.diags(v[augmented_inds])
    else:
        ud, vd = 0.5 * dt * sp.diags(u), 0.5 * dt * sp.diags(v)
    ul = (JDxec@ud + 0.5*JDyec@vd)
    ur = (0.5*JDyec@ud)
    ll = (0.5*JDxec@vd)
    lr = (JDyec@vd + 0.5*JDxec@ud)
    return sp.bmat([[ul, ur], [ll, lr]]) + Eye

def POD(snaps, num_modes=None, method='svd', random_state=None):
    """
    Perform Singular Value Decomposition (SVD) or Randomized SVD (rSVD) to extract POD modes.

    Parameters:
    - snaps       : Snapshot matrix (2D ndarray).
    - num_modes   : Number of modes to compute (int, optional). Default is all.
    - method      : SVD method ('svd' for standard SVD, 'rsvd' for randomized SVD).
    - random_state: Random state for reproducibility when using rSVD.

    Returns:
    - u: Left singular vectors (POD modes).
    - s: Singular values.
    """
    if method == 'svd':
        # Use standard SVD
        u, s, vh = np.linalg.svd(snaps, full_matrices=False)
    elif method == 'rsvd':
        # Use randomized SVD (requires sklearn)
        if num_modes is None:
            num_modes = min(snaps.shape)  # Default to all modes if not provided
        u, s, vh = randomized_svd(snaps, n_components=num_modes, random_state=random_state)
    else:
        raise ValueError("Unknown method '{}' for POD. Use 'svd' or 'rsvd'.".format(method))

    return u, s

def podsize(svals, energy_thresh=None, min_size=None, max_size=None):
    """ Returns the number of vectors in a basis that meets the given criteria """

    if (energy_thresh is None) and (min_size is None) and (max_size is None):
        raise RuntimeError('Must specify at least one truncation criteria in podsize()')

    if energy_thresh is not None:
        svals_squared = np.square(svals.copy())
        energies = np.cumsum(svals_squared)
        energies /= np.square(svals).sum()
        numvecs = np.where(energies >= energy_thresh)[0][0]
    else:
        numvecs = min_size

    if min_size is not None and numvecs < min_size:
        numvecs = min_size

    if max_size is not None and numvecs > max_size:
        numvecs = max_size

    return numvecs

def compute_ECSW_training_matrix_2D(snaps, prev_snaps, basis, res, jac, grid_x, grid_y, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_hdm = int(n_hdm / 2)
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    for isnap in range(n_snaps):
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        ires = res(snap, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        Ji = jac(snap, dt, JDxec, JDyec, Eye)
        Wi = Ji.dot(basis)
        for inode in range(n_hdm):
            C[isnap*n_pod:isnap*n_pod+n_pod, inode] = ires[inode]*Wi[inode] + ires[inode+n_hdm]*Wi[inode+n_hdm]

    return C

def compute_ECSW_training_matrix_2D_rnm(snaps, prev_snaps, basis, approx, jacfwdfunc, res, jac, grid_x, grid_y, dt, mu):
    """
    Assembles the ECSW hyper-reduction training matrix.  Running a non-negative least
    squares algorithm with an early stopping criteria on these matrices will give the
    sample nodes and weights
    This assumes the snapshots are for scalar-valued state variables
    """
    n_hdm, n_snaps = snaps.shape
    n_hdm = int(n_hdm / 2)
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)
    tmu = torch.tensor(mu, dtype=torch.float)
    for isnap in range(n_snaps):#range(1, n_snaps):
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]
        y0 = basis.T @ snap
        y0 = torch.tensor(y0, dtype=torch.float)
        init_res = np.linalg.norm(approx(y0).squeeze().detach().numpy() - snap)
        approx_res = init_res
        num_it = 0
        y = y0.detach()
        print('Initial residual: {:3.2e}'.format(init_res / norm(snap)))
        while abs(approx_res / init_res) > 1e-2 and num_it < 10:
            Jf = jacfwdfunc(y)
            JJ = Jf.T @ Jf
            Jr = Jf.T @ (approx(y) - torch.tensor(snap, dtype=torch.float))
            dy, _, _, _ = np.linalg.lstsq(JJ.squeeze().detach().numpy(), Jr.squeeze().detach().numpy(), rcond=None)
            y -= dy
            approx_res = np.linalg.norm(approx(y).squeeze().detach().numpy() - snap)
            # print('it: {}, Relative residual of fit: {:3.2e}'.format(num_it, abs(approx_res / init_res)))
            num_it += 1
        final_res = np.linalg.norm(approx(y).squeeze().detach().numpy() - snap)
        print('Final residual: {:3.2e}'.format(final_res / norm(snap)))
        ires = res(approx(y).squeeze().detach().numpy(), grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        J = jac(approx(y).squeeze().detach().numpy(), dt, JDxec, JDyec, Eye)
        V = (jacfwdfunc(y).detach()).squeeze().detach().numpy()
        Wi = J.dot(V)
        for inode in range(n_hdm):
            C[(isnap)*n_pod:(isnap)*n_pod+n_pod, inode] = ires[inode]*Wi[inode] + ires[inode+n_hdm]*Wi[inode+n_hdm]

    return C

def compute_ECSW_training_matrix_2D_rbf_nearest_neighbors(snaps, prev_snaps, basis, basis2, epsilon, neighbors,
                                        kdtree, q_p_train, q_s_train, res, jac, grid_x, grid_y, dt, mu, scaler, kernel_type='gaussian'):
    """
    Assembles the ECSW hyper-reduction training matrix for the POD-RBF model.
    Running a non-negative least squares algorithm with an early stopping criteria
    on these matrices will give the sample nodes and weights.
    """
    n_hdm_total, n_snaps = snaps.shape
    n_hdm = n_hdm_total // 2
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))

    # Precompute operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        # Extract current and previous snapshots
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]

        # Initial guess for reduced coordinates q_p by projection
        y0 = basis.T @ snap  # Shape: (n_pod,)

        # Initialize variables for Gauss-Newton iterations
        w_recon = decode_rbf_nearest_neighbors(y0, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)
        init_res = np.linalg.norm(w_recon - snap)
        approx_res = init_res
        num_it = 0
        y = y0.copy()
        print('Initial residual: {:3.2e}'.format(init_res / np.linalg.norm(snap)))

        # Gauss-Newton iterations to refine q_p
        while abs(approx_res / init_res) > 1e-2 and num_it < 10:
            # Compute reconstruction and residual
            w_recon = decode_rbf_nearest_neighbors(y, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)
            res_recon = w_recon - snap  # Residual of reconstruction

            # Compute Jacobian of reconstruction
            Jf = jac_rbf_nearest_neighbors(y, kdtree, q_p_train, q_s_train, basis, basis2, epsilon, neighbors, scaler, kernel_type)

            # Solve for delta y using least squares
            JJ = Jf.T @ Jf
            Jr = Jf.T @ res_recon
            dy, _, _, _ = np.linalg.lstsq(JJ, Jr, rcond=None)
            y -= dy  # Update reduced coordinates

            # Update residual
            w_recon = decode_rbf_nearest_neighbors(y, epsilon, neighbors, kdtree, q_p_train, q_s_train, basis, basis2, scaler, kernel_type)
            approx_res = np.linalg.norm(w_recon - snap)
            num_it += 1

        final_res = np.linalg.norm(w_recon - snap)
        print('Final residual: {:3.2e}'.format(final_res / np.linalg.norm(snap)))

        # Compute residual and Jacobian at the reconstructed state
        ires = res(w_recon, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        J = jac(w_recon, dt, JDxec, JDyec, Eye)

        # Compute the Jacobian of the reconstruction
        V = jac_rbf_nearest_neighbors(y, kdtree, q_p_train, q_s_train, basis, basis2, epsilon, neighbors, scaler, kernel_type)

        # Compute Wi = J * V
        Wi = J @ V

        # Assemble the training matrix
        for inode in range(n_hdm):
            # Rows corresponding to the current snapshot
            row_start = isnap * n_pod
            row_end = row_start + n_pod

            # Contributions from both components (e.g., u and v)
            C[row_start:row_end, inode] = (
                ires[inode] * Wi[inode, :] + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C

def compute_ECSW_training_matrix_2D_rbf_global(snaps, prev_snaps, basis, basis2, W_global, q_p_train, q_s_train, res, jac,
                                               grid_x, grid_y, dt, mu, scaler, epsilon, kernel_type='gaussian'):
    """
    Assembles the ECSW hyper-reduction training matrix for the POD-RBF model using global RBF interpolation.
    Running a non-negative least squares algorithm with an early stopping criterion
    on these matrices will give the sample nodes and weights.

    Parameters:
    - snaps: Snapshots matrix (n_hdm_total x n_snaps).
    - prev_snaps: Previous snapshots for time stepping (n_hdm_total x n_snaps).
    - basis: POD basis (U_p).
    - basis2: Secondary POD basis (U_s).
    - W_global: Precomputed global RBF weights matrix.
    - q_p_train: Training data for primary modes.
    - q_s_train: Training data for secondary modes.
    - res: Residual function for the system.
    - jac: Jacobian function for the system.
    - grid_x, grid_y: Spatial grids.
    - dt: Time step size.
    - mu: List of parameters [mu1, mu2].
    - scaler: MinMaxScaler for normalization.
    - epsilon: RBF parameter.
    - kernel_type: RBF kernel type ('gaussian', 'imq', 'linear', 'multiquadric').

    Returns:
    - C: ECSW training matrix (n_pod * n_snaps x n_hdm).
    """

    n_hdm_total, n_snaps = snaps.shape
    n_hdm = n_hdm_total // 2
    n_pod = basis.shape[1]
    C = np.zeros((n_pod * n_snaps, n_hdm))

    # Precompute operators
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        # Extract current and previous snapshots
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]

        # Initial guess for reduced coordinates q_p by projection
        y0 = basis.T @ snap  # Shape: (n_pod,)

        # Initialize variables for Gauss-Newton iterations
        w_recon = decode_rbf_global(y0, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)
        init_res = np.linalg.norm(w_recon - snap)
        approx_res = init_res
        num_it = 0
        y = y0.copy()
        print(f'Initial residual: {init_res / np.linalg.norm(snap):.2e}')

        # Gauss-Newton iterations to refine q_p
        while abs(approx_res / init_res) > 1e-2 and num_it < 10:
            # Compute reconstruction and residual
            w_recon = decode_rbf_global(y, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)
            res_recon = w_recon - snap  # Residual of reconstruction

            # Compute Jacobian of reconstruction
            Jf = jac_rbf_global(y, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type)

            # Solve for delta y using least squares
            JJ = Jf.T @ Jf
            Jr = Jf.T @ res_recon
            dy, _, _, _ = np.linalg.lstsq(JJ, Jr, rcond=None)
            y -= dy  # Update reduced coordinates

            # Update residual
            w_recon = decode_rbf_global(y, W_global, q_p_train, basis, basis2, epsilon, scaler, kernel_type)
            approx_res = np.linalg.norm(w_recon - snap)
            num_it += 1

        final_res = np.linalg.norm(w_recon - snap)
        print(f'Final residual: {final_res / np.linalg.norm(snap):.2e}')

        # Compute residual and Jacobian at the reconstructed state
        ires = res(w_recon, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        J = jac(w_recon, dt, JDxec, JDyec, Eye)

        # Compute the Jacobian of the reconstruction
        V = jac_rbf_global(y, W_global, q_p_train, q_s_train, basis, basis2, epsilon, scaler, kernel_type)

        # Compute Wi = J * V
        Wi = J @ V

        # Assemble the training matrix
        for inode in range(n_hdm):
            # Rows corresponding to the current snapshot
            row_start = isnap * n_pod
            row_end = row_start + n_pod

            # Contributions from both components (e.g., u and v)
            C[row_start:row_end, inode] = (
                ires[inode] * Wi[inode, :] + ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C

def compute_ECSW_training_matrix_2D_gp(snaps, prev_snaps, basis, basis2,
                                       gp_model,
                                       res, jac,
                                       grid_x, grid_y, dt, mu,
                                       scaler,
                                       max_local_its=10,
                                       local_tol=1e-2):
    """
    Assembles the ECSW hyper-reduction training matrix for the POD-GP model.
    Running a non-negative least squares algorithm with an early stopping
    criterion on these matrices will give the sample nodes and weights.

    Parameters
    ----------
    snaps : ndarray of shape (n_hdm_total, n_snaps)
        Snapshot matrix (HDM space).
    prev_snaps : ndarray of shape (n_hdm_total, n_snaps)
        The previous time-step snapshots (for time marching).
    basis : ndarray of shape (n_hdm_total, r_p)
        Primary POD basis (U_p).
    basis2 : ndarray of shape (n_hdm_total, r_s)
        Secondary POD basis (U_s).
    gp_model : GaussianProcessRegressor (multi-output)
        Trained GP mapping primary coords -> secondary coords.
    res : callable
        Residual function for the system: e.g., `res(w, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)`.
    jac : callable
        Jacobian function for the system: e.g., `jac(w, dt, JDxec, JDyec, Eye)`.
    grid_x, grid_y : ndarray
        Spatial grids in x and y directions.
    dt : float
        Time step size.
    mu : list or tuple of floats
        Parameters [mu1, mu2].
    scaler : MinMaxScaler
        Scaler used to normalize primary POD coords during GP training.
    max_local_its : int, optional
        Maximum Gauss-Newton iterations for local reconstruction refinement. Default is 10.
    local_tol : float, optional
        Tolerance for local reconstruction refinement. Default is 1e-2.

    Returns
    -------
    C : ndarray of shape (n_pod * n_snaps, n_hdm)
        ECSW training matrix used for hyper-reduction.
    """

    n_hdm_total, n_snaps = snaps.shape
    n_hdm = n_hdm_total // 2  # e.g., for 2D (u,v) system
    n_pod = basis.shape[1]

    # The ECSW training matrix
    C = np.zeros((n_pod * n_snaps, n_hdm))

    # Precompute operators (HDM-related)
    Dxec, Dyec, JDxec, JDyec, Eye = get_ops(grid_x, grid_y)

    for isnap in range(n_snaps):
        # Extract current and previous snapshots
        snap = snaps[:, isnap]
        uprev = prev_snaps[:, isnap]

        # Initial guess for reduced coords by projection onto primary basis
        y0 = basis.T @ snap  # shape: (n_pod,)

        # Small Gauss-Newton loop to refine y0 for a better local reconstruction
        w_recon = decode_gp(y0, gp_model, basis, basis2, scaler)
        init_res = np.linalg.norm(w_recon - snap)
        approx_res = init_res
        y = y0.copy()
        num_it = 0

        print(f"Initial reconstruction residual: {init_res / np.linalg.norm(snap):.2e}")

        while abs(approx_res / init_res) > local_tol and num_it < max_local_its:
            w_recon = decode_gp(y, gp_model, basis, basis2, scaler)
            res_recon = w_recon - snap  # how far we are from snapshot

            # Jacobian of the reconstruction (w.r.t. y)
            Jf = jac_gp(y, gp_model, basis, basis2, scaler)

            # Solve linear system Jf^T Jf * dy = Jf^T res_recon (least-squares)
            JJ = Jf.T @ Jf
            Jr = Jf.T @ res_recon
            dy, _, _, _ = np.linalg.lstsq(JJ, Jr, rcond=None)

            y -= dy
            w_recon = decode_gp(y, gp_model, basis, basis2, scaler)
            approx_res = np.linalg.norm(w_recon - snap)
            num_it += 1

        final_res = np.linalg.norm(w_recon - snap)
        print(f"Final reconstruction residual: {final_res / np.linalg.norm(snap):.2e}")

        # Now compute the HDM residual + Jacobian at w_recon
        ires = res(w_recon, grid_x, grid_y, dt, uprev, mu, Dxec, Dyec)
        J_hd = jac(w_recon, dt, JDxec, JDyec, Eye)

        # Compute the Jacobian of the reconstruction for the final y
        V = jac_gp(y, gp_model, basis, basis2, scaler)

        # Wi = J_hd @ V => the linear sensitivity of the HDM residual w.r.t y
        Wi = J_hd @ V

        # Fill the ECSW training matrix
        # "ires" has shape (n_hdm_total,) => we split in the 2D sense
        for inode in range(n_hdm):
            row_start = isnap * n_pod
            row_end = row_start + n_pod

            # Combine contributions from the two state components
            # i.e. u- and v-component, typically found at inode, inode+n_hdm
            C[row_start:row_end, inode] = (
                ires[inode] * Wi[inode, :] +
                ires[inode + n_hdm] * Wi[inode + n_hdm, :]
            )

    return C

def compute_error(rom_snaps, hdm_snaps):
    """ Computes the relative error at each timestep """
    sq_hdm = np.sqrt(np.square(rom_snaps).sum(axis=0))
    sq_err = np.sqrt(np.square(rom_snaps - hdm_snaps).sum(axis=0))
    rel_err = sq_err / sq_hdm
    return rel_err, rel_err.mean()

def param_to_snap_fn(mu, snap_folder="param_snaps", suffix='.npy'):
    """
    Constructs a file path for saving or loading snapshots based on the given parameters (mu).
    
    The file name includes the parameter values concatenated by '+', for example:
    'param_snaps/mu1_4.25+mu2_0.015.npy'.
    
    Parameters:
    - mu: List of parameter values used for generating the snapshot.
    - snap_folder: Folder where snapshots are stored (default is "param_snaps").
    - suffix: File extension for the snapshot file (default is '.npy').

    Returns:
    - snap_fn: The full file path for the snapshot.

    Suggested name change: `generate_snapshot_filename`, `create_snapshot_filepath`, or `construct_param_snapshot_path`
    """
    npar = len(mu)  # Number of parameters
    snapfn = snap_folder + '/'  # Base folder path for the snapshot
    for i in range(npar):
        if i > 0:
            snapfn += '+'  # Use '+' to concatenate multiple parameter values in the filename
        param_str = 'mu{}_{}'.format(i+1, mu[i])  # Format each parameter as mu1_value, mu2_value, etc.
        snapfn += param_str  # Append each formatted parameter to the file path
    return snapfn + suffix  # Add file extension (.npy by default)

def get_saved_params(snap_folder="param_snaps"):
    param_fn_set = set(glob.glob(snap_folder+'/*'))
    return param_fn_set

def load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder="param_snaps"):
    """
    Load snapshots for given parameters if available, otherwise compute and save them.

    Parameters:
    - mu: Parameter vector [mu1, mu2].
    - grid_x, grid_y: Grid arrays in x and y directions.
    - w0: Initial condition for the simulation.
    - dt: Time step.
    - num_steps: Number of steps in the simulation.
    - snap_folder: Directory for storing snapshots.

    Returns:
    - snaps: Snapshot array, either loaded or computed.
    """
    
    # Create folder if it doesn't exist
    if not os.path.exists(snap_folder):
        os.makedirs(snap_folder)
    
    # Generate filename for the snapshot
    snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    
    # Load if the snapshot exists, otherwise compute and save it
    if snap_fn in get_saved_params(snap_folder=snap_folder):
        print(f"Loading saved snaps for mu1={mu[0]}, mu2={mu[1]}")
        snaps = np.load(snap_fn)[:, :num_steps+1]
    else:
        print(f"Computing new snaps for mu1={mu[0]}, mu2={mu[1]}")
        t0 = time.time()
        snaps = inviscid_burgers_implicit2D(grid_x, grid_y, w0, dt, num_steps, mu)
        print('Elapsed time: {:3.3e}'.format(time.time() - t0))
        np.save(snap_fn, snaps)
    
    return snaps

def plot_snaps(grid_x, grid_y, snaps, snaps_to_plot, linewidth=2, color='black', linestyle='solid',
               label=None, fig_ax=None):

    if (fig_ax is None):
        fig, (ax1, ax2) = plt.subplots(2, 1)
    else:
        fig, ax1, ax2 = fig_ax


    x = (grid_x[1:] + grid_x[:-1])/2
    y = (grid_y[1:] + grid_y[:-1])/2
    mid_x = int(x.size / 2)
    mid_y = int(y.size / 2)
    is_first_line = True
    for ind in snaps_to_plot:
        if is_first_line:
            label2 = label
            is_first_line = False
        else:
            label2 = None
        snap = snaps[:(y.size*x.size), ind].reshape(y.size, x.size)
        ax1.plot(x, snap[mid_y, :],
                color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u_x(x,y={:0.1f})$'.format(y[mid_y]))
        #ax1.set_title('$x$-axis $y$-midpoint slice')
        ax1.grid()
        ax2.plot(y, snap[:, mid_x],
                 color=color, linestyle=linestyle, linewidth=linewidth, label=label2)
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u_x(x={:0.1f},y)$'.format(x[mid_x]))
        #ax2.set_title('$y$-axis $x$-midpoint slice')
        ax2.grid()
    return fig, ax1, ax2
