import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hypernet2D import load_or_compute_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

# === Configuration ===
SAMPLES_PER_MU = 3
LATENT_DIM = 10
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-3
SNAP_FOLDER = "../param_snaps"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Snapshot Parameters ===
def get_snapshot_params():
    MU1_RANGE = 4.25, 5.5
    MU2_RANGE = 0.015, 0.03
    mu1_samples = np.linspace(*MU1_RANGE, SAMPLES_PER_MU)
    mu2_samples = np.linspace(*MU2_RANGE, SAMPLES_PER_MU)
    return [[mu1, mu2] for mu1 in mu1_samples for mu2 in mu2_samples]

# === Autoencoder ===
class ScaledAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, mu_in, sig_in):
        super().__init__()
        self.mu_in = torch.tensor(mu_in).float().view(1, -1)
        self.sig_in = torch.tensor(sig_in).float().view(1, -1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def scale_in(self, x):
        return (x - self.mu_in.to(x.device)) / self.sig_in.to(x.device)

    def scale_out(self, y):
        return y * self.sig_in.to(y.device) + self.mu_in.to(y.device)

    def forward(self, x):
        x_scaled = self.scale_in(x)
        z = self.encoder(x_scaled)
        x_recon_scaled = self.decoder(z)
        return self.scale_out(x_recon_scaled)

# === Load Snapshots ===
def load_all_snapshots():
    grid_x, grid_y = GRID_X, GRID_Y
    w0 = W0
    mu_samples = get_snapshot_params()
    first_snap = load_or_compute_snaps(mu_samples[0], grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER)
    snapshot_shape = first_snap.shape
    total_snaps = snapshot_shape[1] * len(mu_samples)
    snaps = np.zeros((snapshot_shape[0], total_snaps))
    col_offset = 0
    for mu in mu_samples:
        try:
            snap_mu = load_or_compute_snaps(mu, grid_x, grid_y, w0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER)
            snaps[:, col_offset:col_offset + snap_mu.shape[1]] = snap_mu
            col_offset += snap_mu.shape[1]
        except FileNotFoundError as e:
            print(f"Missing snapshot for {mu}: {e}")
    return snaps[:, :col_offset]  # shape: (DoFs, snapshots)

# === Training ===
def train():
    snaps = load_all_snapshots()
    snaps = snaps.T  # shape: (snapshots, DoFs)
    mu = snaps.mean(axis=0)
    sigma = snaps.std(axis=0) + 1e-10

    snaps_tensor = torch.tensor(snaps, dtype=torch.float32)
    dataset = TensorDataset(snaps_tensor)

    # 90% train, 10% validation split
    total_len = len(dataset)
    val_len = int(0.1 * total_len)
    train_len = total_len - val_len
    train_data, val_data = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    model = ScaledAutoencoder(input_dim=snaps.shape[1], latent_dim=LATENT_DIM, mu_in=mu, sig_in=sigma).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []
    best_val_loss = float('inf')

    print("Training started...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_rel_error = 0.0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            rel_error = torch.norm(x_recon - x) / (torch.norm(x) + 1e-12)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_rel_error += rel_error.item() * x.size(0)
        avg_train_loss = train_loss / train_len
        avg_train_rel_error = 100 * train_rel_error / train_len

        model.eval()
        val_loss = 0.0
        val_rel_error = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(DEVICE)
                x_recon = model(x)
                loss = criterion(x_recon, x)
                rel_error = torch.norm(x_recon - x) / (torch.norm(x) + 1e-12)
                val_loss += loss.item() * x.size(0)
                val_rel_error += rel_error.item() * x.size(0)
        avg_val_loss = val_loss / val_len
        avg_val_rel_error = 100 * val_rel_error / val_len

        loss_history.append((avg_train_loss, avg_val_loss))
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.6e} ({avg_train_rel_error:.2f}%) - Val Loss: {avg_val_loss:.6e} ({avg_val_rel_error:.2f}%)")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/autoencoder_best.pth")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder_last.pth")
    np.savez("models/scaler.npz", mu=mu, sigma=sigma)

    # Plot losses
    loss_history = np.array(loss_history)
    plt.plot(loss_history[:, 0], label='Train Loss')
    plt.plot(loss_history[:, 1], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("models/loss_curve.png")
    plt.show()

if __name__ == '__main__':
    train()


