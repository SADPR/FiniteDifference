"""
Build a parameterized ROM with a global ROB, and compare it to the HDM at an out-of-sample
point
"""
import numpy as np
import time

import torch
from torch import nn
import torch.optim as optim

from models import RNM_NN  # Ensure RNM_NN is correctly defined in your models.py
from train_utils import get_data, random_split, show_model, TrainingMonitor  # Ensure these are correctly defined
from config import SEED, NUM_CELLS, TRAIN_FRAC, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU, BATCH_SIZE  # Ensure config.py has these variables

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 5000
ROM_SIZE = 40
LR_INIT = 1e-3
LR_PATIENCE = 100
COMPLETION_PATIENCE = 500
MODEL_PATH = 'autoenc_.pt'
CARLBERG = False

from hypernet2D import (load_or_compute_snaps, make_2D_grid, inviscid_burgers_implicit2D_LSPG,
                      plot_snaps, POD)


def train(loader, model, loss_fn, opt, device, verbose=False):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(loader):
        # Convert to float and move to device
        X = X.float().to(device)
        y = y.float().to(device)
        out = model(X)
        loss = loss_fn(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            train_loss += loss.item()

        if verbose:
            if batch % 20 == 0:
                loss_val, current = loss.item(), batch * len(X)
                print("loss: {:.7f}  [{:5d} / {:5d}]".format(loss_val, current, size))
    train_loss /= num_batches
    print("  Train loss: {:.7f}".format(train_loss))
    return train_loss


def test(loader, model, loss_fn, device):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.float().to(device)
            y = y.float().to(device)
            out = model(X)
            test_loss += loss_fn(out, y).item()
    test_loss /= num_batches
    print("  Test loss: {:.7f}".format(test_loss))
    return test_loss


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


def main(num_vecs=10, max_v2=150, compute_basis=False):

    snap_folder = 'param_snaps'

    dt = 0.05
    num_steps = 500
    num_cells_x, num_cells_y = 250, 250
    xl, xu, yl, yu = 0, 100, 0, 100
    grid_x, grid_y = make_2D_grid(xl, xu, yl, yu, num_cells_x, num_cells_y)
    u0 = np.ones((num_cells_y, num_cells_x)).ravel()
    v0 = u0.copy()
    w0 = np.concatenate((u0, v0))

    mu_samples = get_snapshot_params()

    # Compute the basis by collecting snapshots over a range of parameters
    snap_count = len(mu_samples)  # Total number of parameter combinations
    snapshot_shape = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder).shape
    total_snaps = snapshot_shape[1] * snap_count  # Total number of snapshots (time steps * parameter combinations)

    # Pre-allocate memory for all snapshots
    snaps = np.zeros((snapshot_shape[0], total_snaps))

    # Collect snapshots into the pre-allocated array
    col_offset = 0
    for mu in mu_samples:
        snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu  # Insert directly
        col_offset += snap_mu.shape[1]  # Update column offset for the next parameter set

    # Load or compute the HDM snapshots and build the Reduced Order Basis (ROB)
    if not compute_basis:
        # Load a precomputed basis
        basis = np.load('basis.npy', allow_pickle=True)
    else:
        # **Start timing for the SVD computation**
        t0_svd = time.time()

        # Define the method for computing the POD
        pod_method = 'rsvd'  # Choose between 'svd' (standard SVD) or 'rsvd' (randomized SVD)

        # Perform POD to compute the ROB based on the selected method
        if pod_method == 'rsvd':
            # If using randomized SVD, specify the number of modes and optionally a random state
            basis, sigma = POD(snaps, num_modes=max_v2, method='rsvd', random_state=42)
        else:
            # Use standard SVD if no specific method is chosen or 'svd' is selected
            basis, sigma = POD(snaps, num_modes=max_v2, method='svd')

        # **End timing for SVD and print the result**
        t1_svd = time.time()
        print(f"SVD computation time: {t1_svd - t0_svd:.3f} seconds")

        # Save the computed basis and singular values for future use
        np.save('basis', basis)
        np.save('sigma', sigma)

    # Save sizes and adjust for max_v2
    np.save('sizes', [num_vecs, max_v2])
    if basis.shape[1] < max_v2:
        print('WARNING: max_v2 is too large, resetting to maximum possible size')
        max_v2 = basis.shape[1] - 1

    qs = basis.T @ snaps  # Shape: (num_modes, total_snaps)
    q2s = qs[num_vecs:max_v2, :]  # Secondary modes

    del(snaps)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # Force CPU usage as in your original code
    print(f"Using {device} device")

    # Set random seeds for reproducibility
    rng = torch.Generator()
    rng = rng.manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)

    mu_label0 = []
    mu_label1 = []
    for mu in mu_samples:
      vec = mu[0]
      mu_label0.append(np.ones((1, num_steps + 1)) * vec)
      vec = mu[1]
      mu_label1.append(np.ones((1, num_steps + 1)) * vec)
    qs = np.concatenate((qs[:num_vecs, :], np.hstack(mu_label0), np.hstack(mu_label1), qs[num_vecs:, :]))

    # **Prepare Training Data Without Labels**
    train_q, val_q = random_split(qs.T, TRAIN_FRAC, np_rng)
    np.save('train_data', np.vstack([train_q, val_q]))  # Optional: Save training data if needed
    train_t = torch.from_numpy(train_q)
    val_t = torch.from_numpy(val_q)

    # Define input and target tensors
    train_data = TensorDataset(train_t[:, :num_vecs], train_t[:, num_vecs:max_v2])
    val_data = TensorDataset(val_t[:, :num_vecs], val_t[:, num_vecs:max_v2])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)  # Typically, shuffle=False for validation

    # Initialize the neural network
    auto = RNM_NN(num_vecs, max_v2 - num_vecs).to(device)

    # Define loss function and optimizer
    loss = nn.MSELoss()
    opt = optim.Adam(auto.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                     patience=LR_PATIENCE, verbose=True)

    # Initialize Training Monitor
    monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE, auto, opt, scheduler)

    # **Start timing for the neural network training**
    t0_train = time.time()

    # Training Loop
    t = train_loss = test_loss = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}:")
        train_loss = train(train_loader, auto, loss, opt, device)
        test_loss = test(val_loader, auto, loss, device)
        scheduler.step(test_loss)
        if monitor.check_for_completion(train_loss, test_loss):
            break
    print("Training complete!")

    # **End timing for neural network training and print the result**
    t1_train = time.time()
    print(f"Neural network training time: {t1_train - t0_train:.3f} seconds")

    # Plot training curves
    monitor.plot_training_curves()

    return epoch, train_loss, test_loss


if __name__ == "__main__":
    main(compute_basis=True)
