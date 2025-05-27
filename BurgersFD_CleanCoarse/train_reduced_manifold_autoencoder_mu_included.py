"""
Build a parameterized ROM with a global ROB, and compare it to the HDM at an out-of-sample
point
"""

import glob
import pdb

import numpy as np
import matplotlib.pyplot as plt

import sys

import torch
from torch import nn
import torch.optim as optim

from models import RNM_NN
from train_utils import get_data, random_split, show_model, TrainingMonitor
from config import SEED, NUM_CELLS, TRAIN_FRAC, MU1_RANGE, MU2_RANGE, SAMPLES_PER_MU, BATCH_SIZE

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 50000
ROM_SIZE = 40
LR_INIT = 1e-4
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
    # print(X.shape)
    # print(y.shape)
    X = X.float()
    y = y.float()
    X = X.to(device)
    out = model(X)
    loss = loss_fn(out, y)

    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
      train_loss += loss.item()

    if verbose:
      if batch % 20 == 0:
        loss, current = loss.item(), batch * len(X)
        print(  "loss: {:.7f}  [{:5d} / {:5d}]".format(loss, current, size))
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
      X = X.float()
      X = X.to(device)
      out = model(X)
      test_loss += loss_fn(out, y.float()).item()
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

    # Generate or retrive HDM snapshots
    all_snaps_list = []
    for mu in mu_samples:
        snaps = load_or_compute_snaps(mu, grid_x, grid_y, w0, dt, num_steps, snap_folder=snap_folder)
        all_snaps_list += [snaps]

    snaps = np.hstack(all_snaps_list)

    # construct basis using mu_samples params
    if compute_basis:
      basis, sigma = POD(snaps)
      np.save('basis', basis)
      np.save('sigma', sigma)
    else:
      basis = np.load('basis.npy')
    np.save('sizes', [num_vecs, max_v2])
    if basis.shape[1] < max_v2:
        print('WARNING: max_v2 is too large reseting to maximum possible size')
        max_v2 = basis.shape[1] - 1
    qs = basis.T @ snaps
    qs = qs
    q2s = qs[num_vecs:max_v2, :]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using {device} device")

    # torch.set_default_dtype(torch.float32)
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
    
    #qs = (qs - np.mean(qs,axis=0)) / np.std(qs,axis=0)
    train_q, val_q = random_split(qs.T, TRAIN_FRAC, np_rng)
    np.save('train_data', np.vstack([train_q, val_q]))
    train_t = torch.from_numpy(train_q)
    val_t = torch.from_numpy(val_q)
    train_data = TensorDataset(train_t[:, :num_vecs+2], train_t[:, num_vecs+2:max_v2+2])
    val_data = TensorDataset(val_t[:, :num_vecs+2], val_t[:, num_vecs+2:max_v2+2])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    auto = RNM_NN(num_vecs+2, max_v2-num_vecs).to(device)
    # auto = Autoencoder(enc, dec, scaler, unscaler).to(device)
    loss = nn.MSELoss()
    opt = optim.Adam(auto.parameters(), lr=LR_INIT)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                     patience=LR_PATIENCE, verbose=True)

    monitor = TrainingMonitor(MODEL_PATH, COMPLETION_PATIENCE, auto, opt, scheduler)
    if len(sys.argv) > 1:
        monitor.load_from_path(sys.argv[1])
    t = train_loss = test_loss = 0
    for t in range(EPOCHS):
        print("\nEpoch {}:".format(t + 1))
        train_loss = train(train_loader, auto, loss, opt, device)
        test_loss = test(val_loader, auto, loss, device)
        scheduler.step(test_loss)
        if monitor.check_for_completion(train_loss, test_loss):
            break
    print("Training complete!")

    monitor.plot_training_curves()
    # show_model(auto, train_data, val_data, device=device)
    # plt.show()

    # snap_fn = param_to_snap_fn(mu, snap_folder=snap_folder)
    # saved_params = get_saved_params(snap_folder=snap_folder)
    # if snap_fn in saved_params:
    #   print("Loading saved snaps for mu1={}, mu2={}".format(mu[0], mu[1]))
    #   snaps = np.load(snap_fn)[:, :num_steps + 1]
    # else:
    #   snaps = inviscid_burgers_implicit(grid, w0, dt, num_steps, mu)
    #   np.save(snap_fn, snaps)
    np.save('basis', basis)
    return t, train_loss, test_loss


if __name__ == "__main__":
    main(compute_basis=False)
