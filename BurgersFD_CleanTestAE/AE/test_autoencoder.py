import os
import numpy as np
import torch
import torch.nn as nn
import sys

# Ensure the parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from hypernet2D import load_or_compute_snaps
from config import DT, NUM_STEPS, GRID_X, GRID_Y, W0

# === Configuration ===
SNAP_FOLDER = "../param_snaps"
MODEL_DIR = "models"
LATENT_DIM = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Snapshot Parameters ===
def get_snapshot(mu1, mu2):
    return load_or_compute_snaps([mu1, mu2], GRID_X, GRID_Y, W0, DT, NUM_STEPS, snap_folder=SNAP_FOLDER)

# === Autoencoder Class (must match training) ===
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
        z = self.encoder(self.scale_in(x))
        return self.scale_out(self.decoder(z))

# === Evaluation ===
def evaluate(mu1, mu2):
    print(f"Evaluating for mu1 = {mu1}, mu2 = {mu2}")
    snaps = get_snapshot(mu1, mu2)  # shape: (DoFs, num_time_steps)
    snaps = snaps.T  # shape: (num_time_steps, DoFs)

    # Load scaling info
    scaler_path = os.path.join(MODEL_DIR, "scaler.npz")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler file not found. Train model first.")
    with np.load(scaler_path) as data:
        mu = data['mu']
        sigma = data['sigma']

    # Load model
    model = ScaledAutoencoder(input_dim=snaps.shape[1], latent_dim=LATENT_DIM, mu_in=mu, sig_in=sigma).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder_best.pth"), map_location=DEVICE))
    model.eval()

    # Inference
    x_tensor = torch.tensor(snaps, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        x_recon = model(x_tensor).cpu().numpy()

    # Compute relative error
    num = np.linalg.norm(snaps - x_recon)
    denom = np.linalg.norm(snaps)
    rel_error = 100 * num / (denom + 1e-12)

    print(f"Relative error: {rel_error:.2f}%")
    return rel_error

if __name__ == '__main__':
    # Example usage:
    mu1 = 4.75
    mu2 = 0.02
    evaluate(mu1, mu2)
